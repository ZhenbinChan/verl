import sys
import re
import os
import subprocess
import tempfile
from collections import OrderedDict
from subprocess import check_output

sys.path.append('/home/chenzhb/Workspaces/verl/verl/utils/sat_solver')
from code_translator import CodeTranslator


class FOLToPythonConverter:
    """将 FOL 声明和约束转换为可执行的 Z3 Python 代码"""
    
    def __init__(self):
        self.declared_enum_sorts = OrderedDict()
        self.declared_int_sorts = OrderedDict()
        self.declared_functions = OrderedDict()
        self.scoped_list_to_type = {}
    
    def parse_declarations(self, declaration_text):
        """
        解析声明文本
        
        支持格式:
        - people = EnumSort([Vladimir, Wendy])
        - ages = IntSort([1, 2, 3, 4, 5])
        - eats = Function([people, meals] -> [foods])
        - eats = Function([people, meals, foods])  # 最后一个是返回类型
        """
        lines = [line.strip() for line in declaration_text.strip().split('\n') if line.strip()]
        
        for line in lines:
            # 跳过注释
            if line.startswith('#'):
                continue
            
            if '=' not in line:
                continue
            
            var_name = line.split('=')[0].strip()
            declaration = line.split('=', 1)[1].strip()
            
            # 解析 EnumSort
            if declaration.startswith('EnumSort'):
                members = self._extract_members(declaration, 'EnumSort')
                self.declared_enum_sorts[var_name] = members
                self.scoped_list_to_type[var_name] = CodeTranslator.ListValType.ENUM
            
            # 解析 IntSort
            elif declaration.startswith('IntSort'):
                members = self._extract_members(declaration, 'IntSort')
                self.declared_int_sorts[var_name] = members
                self.scoped_list_to_type[var_name] = CodeTranslator.ListValType.INT
            
            # 解析 Function
            elif declaration.startswith('Function'):
                args = self._extract_function_args(declaration)
                self.declared_functions[var_name] = args
    
    def _extract_members(self, declaration, sort_type):
        """从 EnumSort([...]) 或 IntSort([...]) 中提取成员"""
        # 提取括号内的内容
        match = re.search(r'\[(.+?)\]', declaration)
        if match:
            members_str = match.group(1)
            # 分割并清理
            members = [m.strip() for m in members_str.split(',')]
            return members
        return []
    
    def _extract_function_args(self, declaration):
        """
        从 Function 声明中提取参数
        支持两种格式:
        - Function([people, meals] -> [foods])
        - Function([people, meals, foods])
        """
        # 移除 'Function' 和外层括号
        content = declaration[len('Function('):-1].strip()
        
        # 处理带 -> 的格式
        if '->' in content:
            parts = content.split('->')
            input_part = parts[0].strip()
            output_part = parts[1].strip()
            
            # 提取输入参数
            input_match = re.search(r'\[(.+?)\]', input_part)
            inputs = [x.strip() for x in input_match.group(1).split(',')] if input_match else []
            
            # 提取输出类型
            output_match = re.search(r'\[(.+?)\]', output_part)
            outputs = [x.strip() for x in output_match.group(1).split(',')] if output_match else []
            
            return inputs + outputs
        else:
            # 处理简单列表格式
            match = re.search(r'\[(.+?)\]', content)
            if match:
                return [x.strip() for x in match.group(1).split(',')]
        
        return []
    
    def parse_constraints(self, constraints_text):
        """
        解析约束文本（即 FOL Expression），支持多行
        
        约束（constraint）就是你的 FOL 表达式，例如：
        - ForAll([m:meals], eats(Vladimir, m) != eats(Wendy, m))
        - Distinct([m:meals], eats(p, m))
        - Or(eats(p, breakfast) == hot_cakes, eats(p, breakfast) == omelet)
        
        每行可以是一个约束，或者带有注释的约束（用 ::: 分隔注释和表达式）
        """
        lines = constraints_text.strip().split('\n')
        constraints = []  # 存储提取出来的 FOL 表达式
        
        for line in lines:
            line = line.strip()
            # 跳过空行和纯注释
            if not line or line.startswith('#'):
                continue
            
            # 移除行尾注释（如果有 ::: 分隔符，取后面部分作为 FOL 表达式；否则取整行）
            if ':::' in line:
                constraint = line.split(':::')[-1].strip()
            else:
                constraint = line
            
            constraints.append(constraint)
        
        return constraints

    def convert_to_python(self, declaration_text, constraints_text, add_solver_check=True):
        """
        将 Declaration 和 Constraints（FOL Expressions）转换为可执行的 Python 代码
        
        Args:
            declaration_text: 声明文本（EnumSort, IntSort, Function 等）
            constraints_text: 约束文本（FOL 表达式，如 ForAll, Exists, Distinct 等）
            add_solver_check: 是否添加求解器检查代码，默认 True
        
        Returns:
            生成的 Python 代码字符串
            
        说明：
            - Declaration: 定义变量类型和函数签名
            - Constraint/FOL Expression: 逻辑约束条件，会被添加到 Z3 solver 中
        """
        # 解析输入
        self.parse_declarations(declaration_text)
        constraints = self.parse_constraints(constraints_text)  # constraints 就是 FOL 表达式列表
        
        # 翻译声明
        declaration_lines = []
        
        # 翻译 EnumSort
        for name, members in self.declared_enum_sorts.items():
            declaration_lines += CodeTranslator.translate_enum_sort_declaration(name, members)
        
        # 翻译 IntSort
        for name, members in self.declared_int_sorts.items():
            declaration_lines += CodeTranslator.translate_int_sort_declaration(name, members)
        
        # 翻译 List（将 EnumSort 和 IntSort 的成员作为 Python list）
        for name, members in self.declared_enum_sorts.items():
            declaration_lines += CodeTranslator.translate_list_declaration(name, members)
        
        for name, members in self.declared_int_sorts.items():
            declaration_lines += CodeTranslator.translate_list_declaration(name, members)
        
        # 翻译 Function
        for name, args in self.declared_functions.items():
            declaration_lines += CodeTranslator.translate_function_declaration(name, args)
        
        # 翻译约束
        constraint_lines = []
        for constraint in constraints:
            constraint_lines += CodeTranslator.translate_constraint(constraint, self.scoped_list_to_type)
        

        lines = []
        
        # Header
        lines.append("from z3 import *")
        lines.append("")
        
        # Declarations
        for decl_line in declaration_lines:
            lines.append(decl_line.line)
        lines.append("")
        
        # Constraints
        lines.append("# Constraints")
        lines.append("solver = Solver()")
        lines.append("")
        
        for cons_line in constraint_lines:
            if cons_line.line_type == CodeTranslator.LineType.DECL:
                lines.append(cons_line.line)
            else:
                lines.append(f"solver.add({cons_line.line})")
    

        # 添加求解器检查代码（可选）
        if add_solver_check:
            lines.append("")
            lines.append("# Check satisfiability")
            lines.append("if solver.check() == sat:")
            # lines.append("    print('SAT')")
            lines.append("    m = solver.model()")
            lines.append("    print(m)")
            lines.append("    print(1)")
            lines.append("else:")
            lines.append("    print('UNSAT')")
            lines.append("    print(0)")

        
        return "\n".join(lines)
    
    def save_to_file(self, python_code, output_path):
        """Save generated code to file"""
        with open(output_path, 'w') as f:
            f.write(python_code)
    
    def execute_program(self, python_code, timeout=1.0, filter_warnings=True):
        """
        执行生成的 Python 代码，直接返回结果，不保存文件
        
        Args:
            python_code: 生成的 Python 代码字符串
            timeout: 超时时间（秒），默认 1.0 秒
            filter_warnings: 是否过滤警告信息，默认 True
        
        Returns:
            (result, error_message): 
                - result: 执行结果（按行分割的列表），如果失败则返回 None
                - error_message: 错误信息字符串，成功则返回空字符串
        """
        # 使用临时文件执行代码
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            temp_filename = f.name
            # 添加警告过滤器
            if filter_warnings:
                f.write("import warnings\n")
                f.write("warnings.filterwarnings('ignore')\n")
            f.write(python_code)
        
        try:
            # 执行代码，分离 stdout 和 stderr
            process = subprocess.Popen(
                ["python", temp_filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(timeout=timeout)
            
            # 解码输出
            output = stdout.decode("utf-8").strip()
            
            # 如果有标准错误输出且标准输出为空，说明出错了
            if process.returncode != 0:
                error_output = stderr.decode("utf-8").strip()
                error_lines = error_output.splitlines()
                # 获取最后一行错误信息
                error_msg = error_lines[-1] if error_lines else "Unknown Error"
                return None, error_msg
            
            # 分割结果
            result = output.splitlines() if output else []
            
            if len(result) == 0:
                return None, 'No Output'
            
            return result, ""
            
        except subprocess.TimeoutExpired:
            # 超时错误
            process.kill()
            return None, 'TimeoutError'
            
        except Exception as e:
            # 其他错误
            return None, str(e)
            
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_filename)
            except:
                pass


def convert_fol_problem(declaration_text, constraints_text, output_file=None):
    """
    Convert each FOL problem to executable Python code.
    
    Args:
        declaration_text: Declaration 
        constraints_text: FOL Expression/Constraints 
        output_file: output python code path（Optional）
    
    Returns:
        Generated Python Code
    """
    converter = FOLToPythonConverter()
    python_code = converter.convert_to_python(declaration_text, constraints_text)
    
    if output_file:
        converter.save_to_file(python_code, output_file)
        print(f"Code have been save to the path: {output_file}")
    
    return python_code


def convert_and_execute_fol(declaration_text, constraints_text, timeout=1.0):
    """
    转换 FOL 问题为 Python 代码并直接执行，不保存文件
    
    Args:
        declaration_text: Declaration 文本
        constraints_text: FOL Expression/Constraints 文本
        timeout: 执行超时时间（秒）
    
    Returns:
        (python_code, result, error_message):
            - python_code: 生成的 Python 代码
            - result: 执行结果列表（每行一个元素），失败时为 None
            - error_message: 错误信息，成功时为空字符串
    
    Example:
        >>> declarations = "people = EnumSort([Alice, Bob])"
        >>> constraints = "ForAll([p:people], p == Alice)"
        >>> code, result, error = convert_and_execute_fol(declarations, constraints)
        >>> if result:
        >>>     print("\\n".join(result))
    """
    converter = FOLToPythonConverter()
    python_code = converter.convert_to_python(declaration_text, constraints_text)
    result, error_message = converter.execute_program(python_code, timeout)
    
    return python_code, result, error_message


if __name__ == "__main__":
    
    declarations = """
        people = EnumSort([Vladimir, Wendy])
        meals = EnumSort([breakfast, lunch, dinner, snack])
        foods = EnumSort([fish, hot_cakes, macaroni, omelet, poached_eggs])
        eats = Function([people, meals] -> [foods])
    """
    
    constraints = """
        # Constraint 1: At no meal does Vladimir eat the same kind of food as Wendy
        # FOL Expression:
        ForAll([m:meals], eats(Vladimir, m) != eats(Wendy, m))

        # Constraint 2: Neither of them eats the same kind of food more than once during the day
        # FOL Expression:
        ForAll([p:people], Distinct([m:meals], eats(p, m)))

        # Constraint 3: For breakfast, each eats exactly one of the following: hot cakes, poached eggs, or omelets
        # FOL Expression:
        ForAll([p:people], Or(eats(p, breakfast) == hot_cakes, eats(p, breakfast) == poached_eggs, eats(p, breakfast) == omelet))
    """
    
    # 方法1: 转换并保存到文件
    python_code = convert_fol_problem(declarations, constraints, output_file='/home/chenzhb/Workspaces/generated_problem.py')
    
    # 方法2: 转换并直接执行（推荐用于大量问题）
    code, result, error = convert_and_execute_fol(declarations, constraints, timeout=2.0)
    
    if result:
        print("Execute successfully!")
        print("\nExecution result:")
        for line in result:
            print(f"  {line}")
    else:
        print(f"Execute Failed: {error}")

