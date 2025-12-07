from corecode.FileIO import TextFile
from datetime import datetime
from morexai.Configuration import TextToSpeechConfiguration
from pathlib import Path

import hashlib, requests

class TextToSpeechClient:
    """
    Client for XAI Text-to-Speech API.
    """
    BASE_URL = "https://api.x.ai/v1"

    def __init__(self, configuration: TextToSpeechConfiguration):
        self._configuration = configuration
        if self._configuration.base_url is None or \
            self._configuration.base_url == "":
            self._configuration.base_url = TextToSpeechClient.BASE_URL
        if self._configuration.output_directory is None or \
            self._configuration.output_directory == "":
            self._configuration.output_directory = Path.cwd()

        self._api_url = f"{self._configuration.base_url}/audio/speech"

    def _text_file_to_text(self) -> str:
        """
        Convert text file to text.
        """
        if self._configuration.text_file_path is not None:
            return TextFile.load_text(self._configuration.text_file_path)
        return ""

    def _generate_output_filename(self) -> str:
        """
        Generate output filename.
        """
        # Generate hash from current timestamp
        timestamp = datetime.now().isoformat()
        hash_obj = hashlib.sha256(timestamp.encode())
        # Use first 8 characters of hash for brevity
        hash_str = hash_obj.hexdigest()[:8]
        
        # Build filename: prefix_voice_hash.format
        prefix = self._configuration.filename_prefix or "speech"
        voice = self._configuration.voice.lower()
        format_ext = self._configuration.response_format
        
        return f"{prefix}_{voice}_{hash_str}.{format_ext}"

    def text_to_speech(self, xai_api_key: str) -> str:
        """
        Convert text to speech using XAI API.
        """
        text = self._text_file_to_text()
        output_filename = self._generate_output_filename()
        output_path = self._configuration.output_directory / output_filename
        
        # Make API request
        headers = {
            "Authorization": f"Bearer {xai_api_key}",
            "Content-Type": "application/json",
        }
        voice = self._configuration.voice
        response_format = self._configuration.response_format

        print(f"Converting text to speech...")
        print(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"  Voice: {voice}")
        print(f"  Format: {response_format}")
    
        # Make API request
        headers = {
            "Authorization": f"Bearer {xai_api_key}",
            "Content-Type": "application/json",
        }
    
        data = {
            "input": text,
            "voice": voice,
            "response_format": response_format,
        }
    
        try:
            response = requests.post(self._api_url, headers=headers, json=data)
            response.raise_for_status()
            
            # Save audio file
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            print(f"Audio saved to: {output_path}")
            print(f"   Size: {len(response.content)} bytes")
            return str(output_path)
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            if hasattr(e.response, 'text'):
                print(f"   Response: {e.response.text}")
            raise


