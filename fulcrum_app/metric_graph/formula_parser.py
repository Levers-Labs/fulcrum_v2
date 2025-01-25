import re
from typing import List

def parse_formula_references(expression_str: str) -> List[str]:
    """
    Parses a formula expression like "{AcceptOpps} * {SQOToWinRate}"
    and returns a list of references: ["AcceptOpps", "SQOToWinRate"].
    We assume references appear inside curly braces.
    """
    pattern = r"\{([\w]+)\}"
    matches = re.findall(pattern, expression_str)
    return matches