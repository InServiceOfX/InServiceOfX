from dataclasses import dataclass, field
# for string.Formatter
import string
from typing import Any, List, Dict

@dataclass
class FormatString:
    """Wrapper for string formats and string.Formatter."""
    format_string: str
    placeholders: List[str] = field(init=False)
    values: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.placeholders = FormatString.extract_placeholders(
            self.format_string)

    @staticmethod
    def extract_placeholders(format_string: str) -> List[str]:
        formatter = string.Formatter()
        return [
            field[1] for field in formatter.parse(format_string) \
                if field[1] is not None]

    def set_value(self, placeholder: str, value: Any) -> None:
        if placeholder in self.placeholders:
            self.values[placeholder] = value
        else:
            raise ValueError(
                f"Placeholder '{placeholder}' not found in format string.")

    def get_formatted_string(self) -> str:
        """Return format string with only provided values, keeping unfilled
        placeholders intact"""
        format_values = {
            ph: self.values.get(ph, "{" + ph + "}") for ph in self.placeholders}
        return self.format_string.format(**format_values)

    def missing_placeholders(self) -> List[str]:
        return [ph for ph in self.placeholders if ph not in self.values]