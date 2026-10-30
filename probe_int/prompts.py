QUESTION = "{question}\n\nReason step by step and put the final answer in \\boxed{{}}."

PROBE_SUFFIX = "\n</think>\n\nGive only the final answer in \\boxed{}: "

JUDGE_PROBE_SUFFIX = (
    '\n</think>\n\nBased on the reasoning above, is the final answer equal to '
    'the correct answer "{gold_answer}"? Answer only yes or no:'
)

JUDGE_PROBE = """Judge the reasoning prefix, not whether the solver has finished.
Answer yes if a definite mathematical or logical error has occurred and has not been explicitly corrected by the end of the prefix.
Answer no only if the reasoning so far is free of any definite, unresolved error.
Do not answer yes merely because the reasoning is incomplete or uses a different valid method.
If an earlier error is explicitly corrected in the prefix, do not count it as unresolved.
Use the correct answer only as a consistency check; do not judge only by whether it has appeared yet.
Answer only yes or no.

Problem:
{question}

Reference answer:
{gold_answer}

Reasoning so far:
{prefix}"""

ORACLE_SOLVE = """You are an expert mathematician. Solve the problem step by step,
guided by the high-level oracle solution. Show a coherent derivation, but do not
copy the oracle solution verbatim.

<problem>
{question}
</problem>
<high_level_oracle_solution>
{solution}
</high_level_oracle_solution>
"""

PRIVILEGED_SUFFIX_RECONSTRUCTION = """You are an expert mathematician correcting a failed solution.
Continue from the retained prefix and reconstruct the entire remaining reasoning,
including a correct final answer in \\boxed{{}}. Use the reference solution as
privileged guidance. Keep the retained prefix unchanged and do not repeat it.
The original suffix is provided only to identify and avoid its errors. Produce
only the continuation that should follow the retained prefix.

Problem:
{question}

Reference solution (privileged guidance):
{solution}

Retained correct prefix:
{prefix}

Original incorrect suffix to replace:
{suffix}
"""

REPAIR = """You are helping a student correct a mathematical attempt. The
answer probe has already selected the exact segment to replace. Use the
reference solution and the correct solution above to carefully verify this
selected segment against the student's full attempt. Check each claim in the
selected segment, identify its first factual or logical error, and check whether
the student later corrects that error. Do not move the repair to another part
of the attempt.

Propose one intervention step to replace the selected segment. Write it from
the student's perspective so the student can continue naturally from the
retained prefix. Give enough guidance to recover, but do not include the exact
final answer or copy the reference solution.

Draft the intervention, then self-check it: (1) it does not include the exact
final answer, (2) it contains the key insight needed to continue, and (3) it
addresses the selected segment. Revise it until all three checks pass.

Return a detailed verification log, a critical-error report for the first
unrecovered error in the selected segment, and the final intervention. Quote
the relevant student reasoning in the log and report. Use this exact format for
the final intervention so it can be extracted:
<intervention>
{{"content": "ONE reasoning step", "location": 1}}
</intervention>

Problem:
{question}

Reference solution:
{solution}

Correct solution generated above:
{correct_attempt}

Exact final answer for the self-check:
{correct_answer}

Incorrect attempt:
{incorrect_attempt}

Retained prefix immediately before the probe-selected segment:
{prefix}

Probe-selected segment to replace:
{segment}
"""
