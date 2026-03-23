from z3 import *

groups_sort, (Black_Americans, White_Americans, Westernized_Black_Africans, White_Africans) = EnumSort('groups', ['Black_Americans', 'White_Americans', 'Westernized_Black_Africans', 'White_Africans'])
groups = [Black_Americans, White_Americans, Westernized_Black_Africans, White_Africans]
condition = Function('condition', groups_sort, BoolSort())
hypertension_risk = Function('hypertension_risk', groups_sort, IntSort())
salt_content = Function('salt_content', BoolSort(), IntSort())
genetic_adaptation = Function('genetic_adaptation', BoolSort(), BoolSort())

# Constraints
solver = Solver()

solver.add(ForAll(g, Implies([g:group], And(And(Implies((g==Black_Americans), (hypertension_risk(g)==(2*hypertension_risk(White_Americans)))), Implies((g==Westernized_Black_Africans), (hypertension_risk(g)==(2*hypertension_risk(White_Africans))))), Implies((g==Black_Africans), (hypertension_risk(g)==(2*hypertension_risk(White_Africans))))))))

# Check satisfiability
if solver.check() == sat:
    m = solver.model()
    print(m)
    print(1)
else:
    print('UNSAT')
    print(0)