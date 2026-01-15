"""Custom Guardrails validators for input/output validation."""

import re
from typing import Any

from better_profanity import profanity
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="valid_choices", data_type="string")
class ValidChoices(Validator):
    """Validates that a value is one of the allowed choices."""

    def __init__(self, choices: list[str], on_fail: str = "exception"):
        super().__init__(on_fail=on_fail)
        self.choices = choices

    def validate(self, value: Any, metadata: dict = {}) -> ValidationResult:
        if value in self.choices:
            return PassResult()
        return FailResult(
            error_message=f"Value '{value}' is not in allowed choices: {self.choices}"
        )


@register_validator(name="no_profanity", data_type="string")
class NoProfanity(Validator):
    """Validates that input text does not contain profane/toxic words using better-profanity."""

    def __init__(
        self, custom_words: list[str] | None = None, on_fail: str = "exception"
    ):
        super().__init__(on_fail=on_fail)
        if custom_words:
            profanity.add_censor_words(custom_words)

    def validate(self, value: Any, metadata: dict = {}) -> ValidationResult:
        if not profanity.contains_profanity(value):
            return PassResult()
        return FailResult(
            error_message="Input contains prohibited content. Please rephrase."
        )


@register_validator(name="no_pii", data_type="string")
class NoPII(Validator):
    """Detects PII (emails, phone numbers, SSNs) in text."""

    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
        "ssn": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
    }

    def __init__(self, detect: list[str] | None = None, on_fail: str = "exception"):
        super().__init__(on_fail=on_fail)
        self.detect = detect or list(self.PII_PATTERNS.keys())

    def validate(self, value: Any, metadata: dict = {}) -> ValidationResult:
        found_pii = []
        for pii_type in self.detect:
            pattern = self.PII_PATTERNS.get(pii_type)
            if pattern and re.search(pattern, value):
                found_pii.append(pii_type)

        if not found_pii:
            return PassResult()
        return FailResult(
            error_message=f"Input contains PII ({', '.join(found_pii)}). Please remove personal information."
        )


@register_validator(name="no_competitors", data_type="string")
class NoCompetitors(Validator):
    """Blocks mentions of competitor brands or products."""

    def __init__(self, competitors: list[str], on_fail: str = "exception"):
        super().__init__(on_fail=on_fail)
        self.competitors = [c.lower() for c in competitors]

    def validate(self, value: Any, metadata: dict = {}) -> ValidationResult:
        text_lower = value.lower()
        found = [c for c in self.competitors if c in text_lower]
        if not found:
            return PassResult()
        return FailResult(
            error_message="Input mentions competitors. Please focus on our products only."
        )
