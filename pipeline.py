
from utils import *

prompt_dir = "mcts_utils/prompts"
with open(os.path.join(prompt_dir, "Generation1.txt"), "r") as f:
    system_prompt_nl_generation = f.read()
with open(os.path.join(prompt_dir, "Z3DeclarationsGeneration1.txt"), "r") as f:
    system_prompt_z3_declaration = f.read()
with open(os.path.join(prompt_dir, "Z3ImplicationConversion1.txt"), "r") as f:
    system_prompt_z3_implication = f.read()

question_nl = """
<Context>Zhao Ming, Qian Hong and Sun Jie were admitted to Peking University, Tsinghua University and Beijing Normal University. Which schools were they admitted to? The students made the following guesses? Classmate A guessed? Zhao Ming was admitted to Tsinghua University and Sun Jie was admitted to Beijing Normal University. Student B guess? Zhao Ming was admitted to Beijing Normal University, Qian Hong was admitted to Tsinghua University. Student C guess? Zhao Ming was admitted to Peking University, Sun Jie was admitted to Tsinghua University. As a result, the students' guesses were half correct.</Context><Question>Well, their admission status is.</Question><Options>Option (A):Zhao Ming, Qian Hong and Sun Jie were accepted by Peking University, Tsinghua University and Beijing Normal University respectively.

Option (B):Zhao Ming, Qian Hong and Sun Jie were admitted to Tsinghua University, Beijing Normal University and Peking University respectively.

Option (C):Zhao Ming, Qian Hong and Sun Jie were admitted to Beijing Normal University, Tsinghua University and Peking University respectively.

Option (D):Zhao Ming, Qian Hong and Sun Jie were accepted by Peking University, Beijing Normal University and Tsinghua University respectively.</Options>
""".strip()

gold_answer = "A"

answer_nl = get_response(question_nl, system_prompt_nl_generation)
print("="*60)
print(answer_nl)

question_z3 = get_response(question_nl, system_prompt_z3_declaration)
print("="*60)
print(question_z3)

full_input = f"Question:\n{question_nl}\n\nZ3 Declaration:\n{question_z3}\n\nReasoning steps:\n{answer_nl}"
answer_z3 = get_response(full_input, system_prompt_z3_implication)
print("="*60)
print(answer_z3)

parsed_chain = parse_reasoning_steps(answer_nl)
results_nli = verify_steps_nli(parsed_chain)
extracted_answer = extract_final_answer(answer_nl)

extracted_declaration = extract_python_code(question_z3)
extracted_implication = extract_python_code(answer_z3)
prased_implication = parse_python_logic_steps(extracted_implication)
results_fol = verify_steps_fol(extracted_declaration, prased_implication)

for idx, (nl, fol, result_nli, result_fol) in enumerate(zip(parsed_chain, prased_implication, results_nli, results_fol)):
    print(f"Step {idx+1}:")
    print("Premises (NL):")
    for premise in nl['premises']:
        print(f"- {premise}")
    print(f"Conclusion (NL): {nl['conclusion'][0]}")
    print(f"Verification Result (NLI): {result_nli['label']} (Confidence: {result_nli['score']:.4f})")
    print("Premises (FOL):")
    for premise in fol['premises']:
        print(f"- {premise}")
    print(f"Conclusion (FOL): {fol['conclusion'][0]}")
    print(f"Verification Result (FOL): {results_fol[idx]}")
    print("-"*40)

print(f"Extracted Answer: {extracted_answer}")
print(f"Gold Answer: {gold_answer}")
print(f"Answer Correct: {extracted_answer == gold_answer}")