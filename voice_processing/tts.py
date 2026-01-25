"""Text-to-Speech processing using ElevenLabs"""
import os
from typing import Optional


class GoogleTTS:
    """Text-to-Speech using Google TTS (gTTS) - Free fallback"""
    
    def __init__(self):
        """Initialize Google TTS (no API key needed)"""
        try:
            from gtts import gTTS
            self.gTTS = gTTS
        except ImportError:
            raise ImportError("gtts package not installed. Run: pip install gtts")
    
    def generate_speech(self, text: str) -> Optional[bytes]:
        """
        Convert text to speech using Google TTS
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio bytes in MP3 format, or None if error
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            import io
            
            # Create gTTS object
            tts = self.gTTS(text=text, lang='en', slow=False)
            
            # Save to bytes buffer
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            return audio_buffer.read()
            
        except Exception as e:
            raise RuntimeError(f"Google TTS error: {str(e)}")


class ElevenLabsTTS:
    """Text-to-Speech using ElevenLabs API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize ElevenLabs TTS client
        
        Args:
            api_key: ElevenLabs API key (optional, reads from env if not provided)
        """
        self.api_key = api_key or os.getenv('ELEVENLABS_API_KEY')
        if not self.api_key:
            raise ValueError("ElevenLabs API key required. Set ELEVENLABS_API_KEY env variable.")
        
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ElevenLabs SDK client"""
        try:
            from elevenlabs.client import ElevenLabs
            self.client = ElevenLabs(api_key=self.api_key)
        except ImportError:
            raise ImportError("elevenlabs package not installed. Run: pip install elevenlabs")
    
    def generate_speech(
        self, 
        text: str, 
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel voice
        model_id: str = "eleven_turbo_v2_5"  # Fast, free tier compatible
    ) -> Optional[bytes]:
        """
        Convert text to speech audio
        
        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID (default: Rachel)
            model_id: Model to use (default: eleven_turbo_v2_5 for free tier)
            
        Returns:
            Audio bytes in MP3 format, or None if error
        """
        if not self.client:
            raise RuntimeError("ElevenLabs client not initialized")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            
            audio_generator = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_id
            )
            
            # Convert generator to bytes
            audio_bytes = b"".join(audio_generator)
            return audio_bytes
            
        except Exception as e:
            raise RuntimeError(f"ElevenLabs TTS error: {str(e)}")
