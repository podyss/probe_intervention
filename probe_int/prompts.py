QUESTION = "{question}\n\nReason step by step and put the final answer in \\boxed{{}}."

PROBE_SUFFIX = "\n</think>\n\nGive only the final answer in \\boxed{}: "

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

REPAIR = """You are helping a student correct a mathematical attempt. The probe
has already selected the exact segment to replace. Do NOT search for a different
first error or revise any other part of the attempt. Use the oracle solution and
correct reference attempt as privileged information to formulate one useful
intervention, but do not reveal the final answer or copy the reference solution.

Return exactly ONE concise reasoning step from the student's perspective. Do not
include an error report, heading, tags, citations, or final answer. Do not use
blank lines inside the step. The step replaces the selected segment and is
concatenated directly after the retained prefix; continue any partial sentence.

<problem>
{question}
</problem>
<high_level_oracle_solution>
{solution}
</high_level_oracle_solution>
<correct_reference_attempt>
{correct_attempt}
</correct_reference_attempt>
<incorrect_student_attempt>
{incorrect_attempt}
</incorrect_student_attempt>
<retained_prefix>
{prefix}
</retained_prefix>
<suspected_segment_to_replace>
{segment}
</suspected_segment_to_replace>
"""
