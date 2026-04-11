from z3 import *

grades_sort, ('A', 'B', 'C', 'D', 'E') = EnumSort('grades', [''A'', ''B'', ''C'', ''D'', ''E''])
grades = ['A', 'B', 'C', 'D', 'E']
consecutive_grades = Function('consecutive_grades', grades_sort, BoolSort())
john_grades = Function('john_grades', ['economics'_sort, 'geology'_sort, 'history'_sort, 'Italian'_sort, 'physics'_sort, 'Russian'_sort, grades_sort)

# Constraints
solver = Solver()

solver.add(letter_consecutive(grades("A"), grades("B")) == True)
solver.add(letter_consecutive(grades("B"), grades("C")) == True)
solver.add(letter_consecutive(grades("C"), grades("D")) == True)
solver.add(letter_consecutive(grades("D"), grades("E")) == True)
solver.add(letter_consecutive(grades_received(courses("geology")), grades_received(courses("physics"))) == True)
solver.add(letter_consecutive(grades_received(courses("Italian")), grades_received(courses("Russian"))) == True)
solver.add(grades_received(courses("economics")) > grades_received(courses("history")))
solver.add(grades_received(courses("geology")) > grades_received(courses("physics")))
solver.add(grades_received(courses("physics")) > grades_received(courses("Italian")))
solver.add(letter_consecutive(grades_received(courses("physics")), grades_received(courses("Italian"))) == True)
solver.add(grades_received(courses("Russian")) != grades_received(courses("physics")))

# Check satisfiability
if solver.check() == sat:
    m = solver.model()
    print(m)
    print(1)
else:
    print('UNSAT')
    print(0)