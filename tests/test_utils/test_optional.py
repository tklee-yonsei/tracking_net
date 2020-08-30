from typing import Optional
from unittest import TestCase

from utils import optional


class TestOptional(TestCase):
    def test_optional(self):
        a_optional: Optional[str] = "abc"
        a: str = optional.get_or_else(a_optional, "default_abc")
        self.assertEqual(a, "abc")

    def test_optional2(self):
        a_optional: Optional[str] = None
        a: str = optional.get_or_else(a_optional, "default_abc")
        self.assertEqual(a, "default_abc")
