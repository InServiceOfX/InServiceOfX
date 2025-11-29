from dataclasses import dataclass, asdict
from typing import Literal, Optional, Dict, Any, Tuple, List
import hashlib
import json

@dataclass
class UserMessageWithImageURL:
    """Message class for user messages with image URLs (OpenAI Vision API format)"""
    content: List[Dict[str, Any]]
    role: Literal["user"] = "user"

    @classmethod
    def from_text_and_image(cls, text_content: str, image_url: str) \
        -> 'UserMessageWithImageURL':
        """
        Create a UserMessageWithImageURL from text content and an image URL.

        Args:
            text_content: The text prompt/question
            image_url: URL of the image

        Returns:
            UserMessageWithImageURL instance ready for OpenAI Vision API
            
        Example:
            >>> message = UserMessageWithImageURL.from_text_and_image(
            ...     "What's in this image?",
            ...     "https://example.com/image.png"
            ... )
            >>> message.to_dict()
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"}
                    }
                ]
            }
        """
        content = [
            {"type": "text", "text": text_content},
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        ]
        return cls(content=content)

    @staticmethod
    def to_string_and_counts(
        message: 'UserMessageWithImageURL',
        prefix: Optional[str] = None) -> Tuple[str, int, int]:
        """Convert message to string with optional prefix override and return
        character and word counts of the content"""
        role = prefix if prefix else message.role.capitalize()
        
        # Extract text content from the content list
        text_parts = []
        for item in message.content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        
        text_content = " ".join(text_parts)
        formatted = f"{role}: {text_content}"
        char_count = len(text_content)
        word_count = len(text_content.split()) if text_content else 0
        return formatted, char_count, word_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format for OpenAI Vision API
        requests"""
        return {
            "role": self.role,
            "content": self.content
        }

    @staticmethod
    def _hash_content(content: List[Dict[str, Any]]) -> str:
        """
        Generate SHA256 hash of message content.
        
        For content with images, this hashes the text content and image URLs
        in a deterministic way.
        
        Args:
            content: List of content items (text and image_url dicts)
            
        Returns:
            SHA256 hex digest of the content
        """
        # Create a deterministic representation of the content
        # Sort by type to ensure consistent hashing
        content_items = []
        for item in content:
            if item.get("type") == "text":
                content_items.append(("text", item.get("text", "")))
            elif item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                content_items.append(("image_url", url))
        
        # Sort to ensure deterministic hashing (text first, then images)
        content_items.sort(key=lambda x: (x[0] != "text", x[0], x[1]))
        
        # Create a JSON string representation for hashing
        # This ensures consistent hashing regardless of dict key order
        content_str = json.dumps(content_items, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def hash(self) -> str:
        """Generate hash of this message's content"""
        return self._hash_content(self.content)