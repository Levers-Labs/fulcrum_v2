from typing import List

class MockSemanticConnector:
    """
    A mock connector that pretends to be a semantic layer.
    In production, you'd implement an actual connector 
    that queries Cube or dbt for the list of known metrics or members.
    """

    def __init__(self, known_members: List[str]):
        """
        known_members is a list of valid 'member' identifiers
        that the semantic layer recognizes.
        For example: ["metric_1_member", "new_biz_deals", ...]
        """
        self._known_members = set(known_members)

    def member_exists(self, member_name: str) -> bool:
        """
        Returns True if the member_name is recognized in our mock environment.
        """
        return member_name in self._known_members
