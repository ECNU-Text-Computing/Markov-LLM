import math
import sys

def hard_attention(abstract, reference):
    """
    Generate a prompt to assess the importance of a reference paper based on its abstract.
    """
    # Truncate both abstract and reference to the first 1000 characters if longer
    abstract = abstract[:1000] if len(abstract) > 1000 else abstract
    reference = reference[:1000] if len(reference) > 1000 else reference

    prompt = f'''
Determine whether a reference paper is important to a focal paper based on the abstract.
Return Import Index is "1" if it is important and "0" if it is not.
Don't repeat my inputs, just output the values.

Example as follows:
Input:
Focal paper abstract: abstract1
Reference paper abstract: reference1
Output:
0

Input:
Focal paper abstract: {abstract}
Reference paper abstract: {reference}
Output:
'''
    return prompt

def prompt_difference(abstract, reference):
    """
    Generate a prompt to contrast the disruptive potential in the research area of academic papers.
    """
    # Ensure both abstract and reference are truncated to 1000 characters
    abstract = abstract[:1000] if len(abstract) > 1000 else abstract
    reference = reference[:1000] if len(reference) > 1000 else reference

    prompt = f'''
You are now tasked with assessing the disruptive potential in the research area of academic papers.
Your approach involves contrasting the abstract of a focus paper with the abstracts of its cited references.
No need to give me abstract's analysis, just output Contrast and Difference.

Focus paper abstract: {abstract}
Reference paper abstract: {reference}
Contrast and Difference:
'''
    return prompt

def prompt_generation(abstract, reference, d_index=None):
    """
    Generate a prompt to determine if the predicted d-index is high or low.
    """
    # Adjust lengths for abstract and reference to fit within model constraints
    abstract = abstract[:500] if len(abstract) > 500 else abstract
    reference = reference[:1500] if len(reference) > 1500 else reference

    prompt = f'''
- Determine whether the d-index predicted in the previous epoch is high or low: [DINDEX]{d_index}[DINDEX]
- Abstract of Focus Paper: {abstract}
- Comparison with Reference Paper : {reference}
'''
    return prompt

def patent_importance(abstract, reference):
    """
    Generate a prompt to evaluate the importance of a reference patent based on its abstract.
    """
    # Truncate the abstract and reference to manageable lengths
    abstract = abstract[:1000] if len(abstract) > 1000 else abstract
    reference = reference[:1000] if len(reference) > 1000 else reference

    prompt = f'''
Assess the importance of a reference patent based on its abstract in relation to a focal patent.
Return an Importance Index as "1" if it is important and "0" if it is not.
Do not repeat the inputs, only provide the evaluation.

Example as follows:
Input:
Focal Patent Abstract: abstract
Reference Patent Abstract: reference
Output:
0

Input:
Focal Patent Abstract: {abstract}
Reference Patent Abstract: {reference}
Output:
'''
    return prompt

def patent_difference(abstract, reference):
    """
    Generate a prompt to analyze the innovation gap and potential impact between patents.
    """
    # Ensure both abstract and reference are truncated to 1000 characters
    abstract = abstract[:1000] if len(abstract) > 1000 else abstract
    reference = reference[:1000] if len(reference) > 1000 else reference

    prompt = f'''
You are tasked with analyzing the innovation gap and potential impact between patents.
Your job is to contrast the abstract of a focal patent with the abstracts of its related patents.
Avoid providing an analysis of the abstracts themselves; focus instead on the contrast and potential differences.

Focal Patent Abstract: {abstract}
Related Patent Abstract: {reference}
Contrast and Difference:
'''
    return prompt

# Additional functions can be defined following a similar structure and methodology, focusing on creating clear and concise prompts for various analytical tasks.