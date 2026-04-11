import re
import os
from openai import OpenAI
import ast

def get_response(usr_input, system_prompt):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_input},
        ],
        n=1,
        temperature=0.2
    )
    return response.choices[0].message.content


def parse_reasoning_steps(text: str):
    step_pattern = re.compile(r'<step>(.*?)</step>', re.DOTALL)
    steps = step_pattern.findall(text)
    
    parsed_chain = []
    
    for idx, step_content in enumerate(steps):
        premise_pattern = re.compile(r'<premise>(.*?)</premise>', re.DOTALL)
        premises = premise_pattern.findall(step_content)
        cleaned_premises = [p.strip() for p in premises]
        
        conclusion_pattern = re.compile(r'<conclusion>(.*?)</conclusion>', re.DOTALL)
        conclusion_match = conclusion_pattern.search(step_content)
        conclusion = [conclusion_match.group(1).strip() if conclusion_match else ""]
    
        parsed_chain.append({
            "step_index": idx,
            "premises": cleaned_premises,
            "conclusion": conclusion
        })
        
    return parsed_chain


def parse_python_logic_steps(code_str: str):
    tree = ast.parse(code_str)
    raw_data = {}

    # 1. 第一次遍历：提取所有原始赋值（保持顺序独立性）
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign): continue
        
        for target in node.targets:
            if not isinstance(target, ast.Name): continue
            
            name = target.id
            if name.startswith(('premises_', 'conclusion_')):
                ptype, idx = name.split('_')[0], int(name.split('_')[-1])
                raw_data.setdefault(idx, {"premises": [], "conclusion": []})
                
                # 统一提取表达式：如果是 List 则展开，否则包裹为 List
                if ptype == 'premises' and isinstance(node.value, ast.List):
                    raw_data[idx]["premises"] = [ast.unparse(e) for e in node.value.elts]
                else:
                    raw_data[idx][ptype] = [ast.unparse(node.value)]

    # 2. 第二次处理：解引用引用逻辑（将 conclusion_x 替换为具体内容）
    for idx in sorted(raw_data.keys()):
        new_premises = []
        for p in raw_data[idx]["premises"]:
            p_strip = p.strip()
            if p_strip.startswith("conclusion_"):
                ref_idx = int(p_strip.split('_')[-1])
                new_premises.append(raw_data[ref_idx]["conclusion"][0])
            else:
                new_premises.append(p)
        raw_data[idx]["premises"] = new_premises

    # 3. 构造最终输出
    return [
        {
        "step_index": i,
        "premises": raw_data[i]["premises"],
        "conclusion": raw_data[i]["conclusion"]
        } 
        for i in sorted(raw_data.keys())
    ]


def extract_final_answer(response):
    response = response.strip().split('\n')
    resp_text = [x for x in response if x.strip()]
    resp_text = "\n".join(resp_text[-3:])

    answer = None
    if "\\box" in resp_text or "\\boxed" in resp_text:
        answer = re.findall(r'\\box\{([^{}]*(?:\{[^{}]*\})*[^{}]*)\}', resp_text)
        if len(answer) == 0:
            answer = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\})*[^{}]*)\}', resp_text)

    return answer[0].strip() if answer else response.strip()


def extract_python_code(text: str):
    pattern = re.compile(r'```python\n(.*?)\n```', re.DOTALL)
    code_blocks = pattern.findall(text)
    return "\n\n".join(code_blocks) if code_blocks else text.strip()


def verify_steps_nli(parsed_steps):
    import transformers
    classifier = transformers.pipeline("text-classification", model="microsoft/deberta-v2-xlarge-mnli", device=0)
    nli_inputs = []
    for step in parsed_steps:
        nli_inputs.append({
            "text": " ".join(step['premises']),
            "text_pair": step['conclusion'][0]
        })
    results = classifier(nli_inputs)
    return results


def verify_steps_fol(declarations, parsed_steps):
    results = []
    for step in parsed_steps:
        premises_str = ", ".join(step['premises']) if step['premises'] else "True"
        conclusion_str = step['conclusion'][0]
        z3_code = f"""
{declarations}
solver = Solver()
solver.add(And({premises_str}))
solver.add(Not({conclusion_str}))

check_res = solver.check()
if check_res == unsat:
    print("SUCCESS_ENTAILED")
elif check_res == sat:
    print("FAILED_CONTRADICT")
else:
    print("UNKNOWN")
""".strip()
        result, error = execute_program(z3_code)
        if result is None:
            breakpoint()
        results.append(result)
    return results

def execute_program(python_code, timeout=1.0, filter_warnings=True):
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        temp_filename = f.name
        if filter_warnings:
            f.write("import warnings\n")
            f.write("warnings.filterwarnings('ignore')\n")
        f.write(python_code)
    
    try:
        process = subprocess.Popen(
            ["python", temp_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(timeout=timeout)
        output = stdout.decode("utf-8").strip()
        
        if process.returncode != 0:
            error_output = stderr.decode("utf-8").strip()
            error_lines = error_output.splitlines()
            error_msg = error_lines[-1] if error_lines else "Unknown Error"
            return None, error_msg
        
        result = output.splitlines() if output else []
        
        if len(result) == 0:
            return None, 'No Output'
        
        return result, ""
        
    except subprocess.TimeoutExpired:
        process.kill()
        return None, 'TimeoutError'
        
    except Exception as e:
        return None, str(e)
        
    finally:
        try:
            os.unlink(temp_filename)
        except:
            pass
