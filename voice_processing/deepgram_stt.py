"""Speech-to-Text using Deepgram SDK"""
import os
from typing import Optional
from deepgram import DeepgramClient


class DeepgramSTT:
    """Speech-to-Text using Deepgram SDK"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Deepgram STT client
        
        Args:
            api_key: Deepgram API key (optional, reads from env if not provided)
        """
        self.api_key = api_key or os.getenv('DEEPGRAM_API_KEY')
        if not self.api_key:
            raise ValueError("Deepgram API key required. Set DEEPGRAM_API_KEY env variable.")
        
        # Initialize Deepgram SDK client
        self.client = DeepgramClient(api_key=self.api_key)
    
    def transcribe_audio(
        self,
        audio_bytes: bytes,
        model: str = "nova-2",
        language: str = "en-US",
        punctuate: bool = True,
        smart_format: bool = True
    ) -> Optional[str]:
        """
        Transcribe audio bytes to text using Deepgram SDK
        
        Args:
            audio_bytes: Raw audio data in bytes
            model: Deepgram model to use (default: nova-2)
            language: Language code (default: en-US)
            punctuate: Add punctuation to transcript
            smart_format: Apply smart formatting
            
        Returns:
            Transcribed text or None if error
        """
        if not audio_bytes:
            raise ValueError("Audio bytes cannot be empty")
        
        try:
           
            response = self.client.listen.v1.media.transcribe_file(
                request=audio_bytes,
                model=model,
                language=language,
                punctuate=punctuate,
                smart_format=smart_format
            )
            
            # Extract transcript from response
            if response.results and response.results.channels:
                channels = response.results.channels
                if len(channels) > 0 and channels[0].alternatives:
                    alternatives = channels[0].alternatives
                    if len(alternatives) > 0:
                        transcript = alternatives[0].transcript
                        return transcript.strip() if transcript else None
            
            return None
            
        except Exception as e:
            raise RuntimeError(f"Deepgram SDK transcription error: {str(e)}")
    
   # def transcribe_file(self, file_path: str, **kwargs) -> Optional[str]:
    #    """
     #   Transcribe audio file to text using Deepgram SDK
      #  
       # Args:
        #    file_path: Path to audio file
         #   **kwargs: Additional parameters for transcribe_audio
          #  
        #Returns:
         #   Transcribed text or None if error
        #"""
        #try:
         #   with open(file_path, 'rb') as audio_file:
          #      audio_bytes = audio_file.read()
           # return self.transcribe_audio(audio_bytes, **kwargs)
        #except FileNotFoundError:
        #    raise FileNotFoundError(f"Audio file not found: {file_path}")
        #except Exception as e:
        #    raise RuntimeError(f"Error reading audio file: {str(e)}")


# Convenience function for quick usage
def speech_to_text(audio_bytes: bytes, api_key: Optional[str] = None) -> Optional[str]:
    """
    Quick helper to transcribe audio bytes using Deepgram SDK
    
    Args:
        audio_bytes: Raw audio data
        api_key: Optional API key
        
    Returns:
        Transcribed text or None
    """
    stt = DeepgramSTT(api_key=api_key)
    return stt.transcribe_audio(audio_bytes)
