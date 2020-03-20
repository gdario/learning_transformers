import pickle

PICKLE_FILE = 'data/onekm/aurora_cases.pkl'
with open(PICKLE_FILE, 'rb') as fh:
    aurora_cases = pickle.load(fh)

print(len(aurora_cases))

# Length of the individual entries
problem_description = aurora_cases.problem_description.values
evaluation_summary = aurora_cases.evaluation_summary.values
case_summary = aurora_cases.case_summary.values
