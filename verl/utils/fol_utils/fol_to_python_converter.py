"""
FOL to Python (Z3) code converter.

Adapted from verl-binlp/mcts_utils/fol_to_python_converter.py.
Converts FOL declarations and constraints into executable Z3 Python code,
then runs the code in a subprocess to check satisfiability.
"""

import re
import os
import subprocess
import tempfile
from collections import OrderedDict

from .sat_solver.code_translator import CodeTranslator


class FOLToPythonConverter:
    """Convert FOL declarations and constraints to executable Z3 Python code."""

    def __init__(self):
        self.declared_enum_sorts = OrderedDict()
        self.declared_int_sorts = OrderedDict()
        self.declared_functions = OrderedDict()
        self.scoped_list_to_type = {}

    def parse_declarations(self, declaration_text):
        declaration_text = re.sub(r'\btrue\b', 'True', declaration_text)
        declaration_text = re.sub(r'\bfalse\b', 'False', declaration_text)
        lines = [line.strip() for line in declaration_text.strip().split('\n') if line.strip()]

        for line in lines:
            if line.startswith('#'):
                continue
            if '=' not in line:
                continue

            var_name = line.split('=')[0].strip()
            declaration = line.split('=', 1)[1].strip()

            if declaration.startswith('EnumSort'):
                members = self._extract_members(declaration, 'EnumSort')
                self.declared_enum_sorts[var_name] = members
                self.scoped_list_to_type[var_name] = CodeTranslator.ListValType.ENUM
            elif declaration.startswith('IntSort'):
                members = self._extract_members(declaration, 'IntSort')
                self.declared_int_sorts[var_name] = members
                self.scoped_list_to_type[var_name] = CodeTranslator.ListValType.INT
            elif declaration.startswith('Function'):
                args = self._extract_function_args(declaration)
                self.declared_functions[var_name] = args

    def _extract_members(self, declaration, sort_type):
        match = re.search(r'\[(.+?)\]', declaration)
        if match:
            members_str = match.group(1)
            members = [m.strip() for m in members_str.split(',')]
            return members
        return []

    def _extract_function_args(self, declaration):
        content = declaration[len('Function('):-1].strip()
        if '->' in content:
            parts = content.split('->')
            input_part = parts[0].strip()
            output_part = parts[1].strip()
            input_match = re.search(r'\[(.+?)\]', input_part)
            inputs = [x.strip() for x in input_match.group(1).split(',')] if input_match else []
            output_match = re.search(r'\[(.+?)\]', output_part)
            outputs = [x.strip() for x in output_match.group(1).split(',')] if output_match else []
            return inputs + outputs
        else:
            match = re.search(r'\[(.+?)\]', content)
            if match:
                return [x.strip() for x in match.group(1).split(',')]
        return []

    def parse_constraints(self, constraints_text):
        constraints_text = re.sub(r'\btrue\b', 'True', constraints_text)
        constraints_text = re.sub(r'\bfalse\b', 'False', constraints_text)
        lines = constraints_text.strip().split('\n')
        constraints = []

        for line in lines:
            flag = line.startswith(' ')
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':::' in line:
                constraint = line.split(':::')[-1].strip()
            else:
                constraint = line
            if flag and constraints:
                constraints[-1] += ' ' + constraint
            else:
                constraints.append(constraint)

        return constraints

    def convert_to_python(self, declaration_text, constraints_text, add_solver_check=True):
        self.parse_declarations(declaration_text)
        constraints = self.parse_constraints(constraints_text)

        declaration_lines = []
        for name, members in self.declared_enum_sorts.items():
            declaration_lines += CodeTranslator.translate_enum_sort_declaration(name, members)
        for name, members in self.declared_int_sorts.items():
            declaration_lines += CodeTranslator.translate_int_sort_declaration(name, members)
        for name, members in self.declared_enum_sorts.items():
            declaration_lines += CodeTranslator.translate_list_declaration(name, members)
        for name, members in self.declared_int_sorts.items():
            declaration_lines += CodeTranslator.translate_list_declaration(name, members)
        for name, args in self.declared_functions.items():
            declaration_lines += CodeTranslator.translate_function_declaration(name, args)

        constraint_lines = []
        for constraint in constraints:
            constraint_lines += CodeTranslator.translate_constraint(constraint, self.scoped_list_to_type)

        lines = []
        lines.append("from z3 import *")
        lines.append("")
        for decl_line in declaration_lines:
            lines.append(decl_line.line)
        lines.append("")
        lines.append("# Constraints")
        lines.append("solver = Solver()")
        lines.append("")
        for cons_line in constraint_lines:
            if cons_line.line_type == CodeTranslator.LineType.DECL:
                lines.append(cons_line.line)
            else:
                lines.append(f"solver.add({cons_line.line})")

        if add_solver_check:
            lines.append("")
            lines.append("# Check satisfiability")
            lines.append("if solver.check() == sat:")
            lines.append("    m = solver.model()")
            lines.append("    print(m)")
            lines.append("    print(1)")
            lines.append("else:")
            lines.append("    print('UNSAT')")
            lines.append("    print(0)")

        return "\n".join(lines)

    def execute_program(self, python_code, timeout=1.0, filter_warnings=True):
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
                stderr=subprocess.PIPE,
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
            except OSError:
                pass


def convert_and_execute_fol(declaration_text, constraints_text, timeout=1.0):
    """Convert FOL to Z3 Python code and execute it.

    Returns:
        (python_code, result, error_message)
    """
    converter = FOLToPythonConverter()
    try:
        python_code = converter.convert_to_python(declaration_text, constraints_text)
        result, error_message = converter.execute_program(python_code, timeout)
    except Exception as e:
        return None, None, str(e)
    return python_code, result, error_message
