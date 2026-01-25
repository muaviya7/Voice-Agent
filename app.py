import streamlit as st
from rag.retriever import RAGRetriever
from rag.llm_handler import LLMHandler
from voice_processing.tts import ElevenLabsTTS, GoogleTTS
from voice_processing.deepgram_stt import DeepgramSTT
from audio_recorder_streamlit import audio_recorder
import os
import asyncio
from dotenv import load_dotenv
import time
load_dotenv()


def initialize_session():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'retriever' not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            st.session_state.retriever = RAGRetriever()
    
    if 'llm' not in st.session_state:
        api_key = os.getenv('GOOGLE_API_KEY', 'GEMINI_API_KEY_NOT_SET')
        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        with st.spinner("Initializing AI with conversation memory..."):
            st.session_state.llm = LLMHandler(api_key, openrouter_key=openrouter_key)
            # Set up the retriever for conversational chain
            langchain_retriever = st.session_state.retriever.as_langchain_retriever(top_k=5)
            st.session_state.llm.set_retriever(langchain_retriever)
    
    if 'voice_transcript' not in st.session_state:
        st.session_state.voice_transcript = ""
    
    if 'last_audio_hash' not in st.session_state:
        st.session_state.last_audio_hash = None
    
    # Initialize voice processing clients
    if 'tts_client' not in st.session_state:
        try:
            st.session_state.tts_client = ElevenLabsTTS()
        except Exception as e:
            st.session_state.tts_client = None
            print(f"TTS initialization failed: {e}")
    
    if 'google_tts_client' not in st.session_state:
        try:
            st.session_state.google_tts_client = GoogleTTS()
        except Exception as e:
            st.session_state.google_tts_client = None
            print(f"Google TTS initialization failed: {e}")
    
    if 'stt_client' not in st.session_state:
        try:
            st.session_state.stt_client = DeepgramSTT()
        except Exception as e:
            st.session_state.stt_client = None
            print(f"STT initialization failed: {e}")


def display_message(role, content, sources=None, audio_bytes=None):
    """Display a chat message with optional sources and preloaded audio"""
    with st.chat_message(role):
        st.markdown(content)
        
        # Add TTS audio player for assistant responses (audio is preloaded)
        if role == "assistant" and audio_bytes:
            if st.button("🔊 Play Audio", key=f"tts_{hash(content)}", type="secondary", use_container_width=False):
                st.audio(audio_bytes, format="audio/mpeg", autoplay=True)
        
        if sources and role == "assistant":
            with st.expander("📚 View Sources"):
                for i, source in enumerate(sources, 1):
                    relevance = source['relevance'] * 100
                    st.caption(
                        f"{i}. **{source['title']}** "
                        f"({source['category']}) - {relevance:.0f}% relevant"
                    )


def display_multi_model_message(role, multi_content, timestamp=None):
    """Display multi-model responses in 3 columns with timing"""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(multi_content)
        return
    
    # Use current timestamp if none provided (fallback)
    if timestamp is None:
        timestamp = int(time.time())
    
    # Assistant responses in 3 columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.chat_message("assistant"):
            st.markdown("**🤖 Gemini**")
            if 'response_time' in multi_content['gemini']:
                st.caption(f"⏱️ {multi_content['gemini']['response_time']}")
            st.markdown(multi_content['gemini']['answer'])
            if multi_content['gemini'].get('audio_bytes'):
                if st.button("🔊 Play Audio", key=f"gemini_{timestamp}_{hash(multi_content['gemini']['answer'][:50])}", use_container_width=True):
                    st.audio(multi_content['gemini']['audio_bytes'], format="audio/mpeg", autoplay=True)
            if multi_content['gemini'].get('sources'):
                with st.expander("📚 Sources"):
                    for i, s in enumerate(multi_content['gemini']['sources'], 1):
                        st.caption(f"{i}. {s['title']} ({s['relevance']*100:.0f}%)")
    
    with col2:
        with st.chat_message("assistant"):
            st.markdown("**🧠 DeepSeek**")
            if 'response_time' in multi_content['deepseek']:
                st.caption(f"⏱️ {multi_content['deepseek']['response_time']}")
            st.markdown(multi_content['deepseek']['answer'])
            if multi_content['deepseek'].get('audio_bytes'):
                if st.button("🔊 Play Audio", key=f"deepseek_{timestamp}_{hash(multi_content['deepseek']['answer'][:50])}", use_container_width=True):
                    st.audio(multi_content['deepseek']['audio_bytes'], format="audio/mpeg", autoplay=True)
            if multi_content['deepseek'].get('sources'):
                with st.expander("📚 Sources"):
                    for i, s in enumerate(multi_content['deepseek']['sources'], 1):
                        st.caption(f"{i}. {s['title']} ({s['relevance']*100:.0f}%)")
    
    with col3:
        with st.chat_message("assistant"):
            st.markdown("**🔥 Mistral**")
            if 'response_time' in multi_content['mistral']:
                st.caption(f"⏱️ {multi_content['mistral']['response_time']}")
            st.markdown(multi_content['mistral']['answer'])
            if multi_content['mistral'].get('audio_bytes'):
                if st.button("🔊 Play Audio", key=f"mistral_{timestamp}_{hash(multi_content['mistral']['answer'][:50])}", use_container_width=True):
                    st.audio(multi_content['mistral']['audio_bytes'], format="audio/mpeg", autoplay=True)
            if multi_content['mistral'].get('sources'):
                with st.expander("📚 Sources"):
                    for i, s in enumerate(multi_content['mistral']['sources'], 1):
                        st.caption(f"{i}. {s['title']} ({s['relevance']*100:.0f}%)")


def generate_tts(text):
    """
    Generate speech audio from text using ElevenLabs with Google TTS fallback
    
    Args:
        text: Text to convert to speech
        
    Returns:
        Audio bytes or None if error
    """
    try:
        if not st.session_state.tts_client:
            st.error("TTS client not initialized")
            return None
        
        audio_bytes = st.session_state.tts_client.generate_speech(text)
        return audio_bytes
            
    except Exception as e:
        # Fallback to Google TTS if ElevenLabs fails (quota exceeded, etc.)
        print(f"ElevenLabs TTS Error: {str(e)}")
        try:
            if st.session_state.google_tts_client:
                print("Falling back to Google TTS...")
                audio_bytes = st.session_state.google_tts_client.generate_speech(text)
                return audio_bytes
            else:
                st.error(f"TTS Error: {str(e)}")
                return None
        except Exception as fallback_error:
            st.error(f"Both TTS services failed: {str(fallback_error)}")
            return None


def process_query(query, top_k):
    """
    Process user query and generate response with conversation memory
    
    This function:
    1. Uses LangChain ConversationalRetrievalChain
    2. Automatically retrieves relevant context from ChromaDB
    3. Includes conversation history for context-aware responses
    4. Generates TTS audio preloaded with the response
    5. Returns answer with source documents and audio
    
    Args:
        query (str): User's question
        top_k (int): Number of sources to retrieve (controlled by retriever config)
        
    Returns:
        tuple: (answer: str, sources: list[dict], audio_bytes: bytes)
            - answer: Generated response from Gemini with conversation context
            - sources: List of source documents with metadata
            - audio_bytes: Preloaded TTS audio (ready to play)
    """
    # Generate answer using conversational chain (handles RAG + memory automatically)
    with st.spinner("Thinking..."):
        answer, source_docs = st.session_state.llm.generate_answer_with_memory(query)
    
    # Format sources from LangChain documents
    sources = []
    for i, doc in enumerate(source_docs):
        # Note: LangChain's ConversationalRetrievalChain doesn't preserve distance scores
        # Relevance is estimated based on retrieval order (first = most relevant)
        relevance = 1.0 - (i * 0.1)  # 1.0, 0.9, 0.8, 0.7, 0.6...
        
        sources.append({
            'title': doc.metadata.get('title', 'Unknown'),
            'category': doc.metadata.get('category', 'Unknown'),
            'relevance': max(0.5, relevance)  # Minimum 0.5 relevance
        })
    
    # Generate TTS audio (preload so it plays instantly when user clicks)
    audio_bytes = None
    if st.session_state.tts_client:
        with st.spinner("🎵 Preparing audio..."):
            try:
                audio_bytes = generate_tts(answer)
            except Exception as e:
                print(f"TTS generation failed: {e}")
    
    return answer, sources, audio_bytes


async def process_multi_query(query, top_k):
    """
    Process query with all 3 models in parallel
    
    Returns:
        dict: {'gemini': {...}, 'deepseek': {...}, 'phi3': {...}}
    """
    with st.spinner("🤖 Getting responses from all models..."):
        results = await st.session_state.llm.generate_multi_model_answers(query)
    
    # Format sources for each model
    formatted_results = {}
    for model_name, (answer, source_docs, response_time) in results.items():
        sources = []
        for i, doc in enumerate(source_docs):
            relevance = 1.0 - (i * 0.1)
            sources.append({
                'title': doc.metadata.get('title', 'Unknown'),
                'category': doc.metadata.get('category', 'Unknown'),
                'relevance': max(0.5, relevance)
            })
        
        # Generate TTS audio - SAME AS ORIGINAL WORKING METHOD
        audio_bytes = None
        if st.session_state.tts_client:
            try:
                audio_bytes = generate_tts(answer)  # Direct call like original
                print(f"TTS generated for {model_name}: {bool(audio_bytes)}")
            except Exception as e:
                print(f"TTS generation failed for {model_name}: {e}")
        
        formatted_results[model_name] = {
            'answer': answer,
            'sources': sources,
            'audio_bytes': audio_bytes,
            'response_time': response_time
        }
    
    return formatted_results


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # RAG configuration
        st.subheader("🔍 Search Settings")
        top_k = st.slider(
            "Number of sources to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="More sources = more context but slower"
        )
        
        # Actions
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.voice_transcript = ""
            # Clear LangChain conversation memory
            if hasattr(st.session_state.llm, 'memory'):
                st.session_state.llm.memory.clear()
            st.rerun()
        
        # Stats
        st.divider()
        st.caption(f"💬 Messages: {len(st.session_state.messages)}")
        
        return top_k


def render_chat_history():
    """Render all messages in chat history"""
    for msg in st.session_state.messages:
        if 'multi_model' in msg and msg['multi_model']:
            display_multi_model_message(msg["role"], msg["content"], msg.get("timestamp"))
        else:
            display_message(
                role=msg["role"],
                content=msg["content"],
                sources=msg.get("sources"),
                audio_bytes=msg.get("audio_bytes")
            )


def main():
    # Page config
    st.set_page_config(
        page_title="Sunmarke School AI Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    initialize_session()
    
    # Header
    st.title("🎓 Sunmarke School")
    st.caption("AI-Powered Assistant - Ask anything about our school")
    
    # Sidebar settings
    top_k = render_sidebar()
    
    # Multi-model mode is now default (no toggle needed)
    use_multi_model = True  # Always use multi-model mode
    
    # Render chat history first (scrolls up for older messages)
    render_chat_history()
    
    # Fixed bottom input area - SEPARATED from transcription
    st.markdown("---")
    
    # Container for bottom input
    with st.container():
        # Handle voice transcript first (preserve existing logic)
        if st.session_state.voice_transcript:
            if "user_query_input" not in st.session_state:
                st.session_state.user_query_input = ""
            st.session_state.user_query_input = st.session_state.voice_transcript
            st.session_state.voice_transcript = ""  # Clear after moving
        
        # Create layout: Text Input | Recording Button | Send Button
        input_col1, input_col2, input_col3 = st.columns([6, 1, 1])
        
        with input_col1:
            user_input = st.text_input(
                "Message",
                placeholder="Ask about admissions, curriculum, activities...",
                label_visibility="collapsed",
                key="user_query_input"
            )
        
        with input_col2:
            # Voice transcription processing (SEPARATE from input to avoid re-runs)
            if st.session_state.stt_client:
                audio_bytes = audio_recorder(
                    text="🎤",
                    recording_color="#e74c3c",
                    neutral_color="#3498db",
                    icon_size="1x",
                    energy_threshold=(-1.0, 1.0),
                    pause_threshold=30.0
                )
                
                # Process audio (preserve existing transcription logic)
                if audio_bytes:
                    audio_hash = hash(audio_bytes)
                    if audio_hash != st.session_state.last_audio_hash:
                        with st.spinner("🔄 Transcribing..."):
                            try:
                                transcript = st.session_state.stt_client.transcribe_audio(audio_bytes)
                                if transcript:
                                    st.session_state.voice_transcript = transcript
                                    st.session_state.last_audio_hash = audio_hash
                                    st.success(f"📝 {transcript}")
                                    st.rerun()
                                else:
                                    st.error("No speech detected")
                            except Exception as e:
                                st.error(f"Transcription Error: {str(e)}")
        
        with input_col3:
            send_button = st.button("📤 Send", use_container_width=True, type="primary")
    
    # Process query ONLY when send button is clicked
    if send_button and user_input:
        # Add user message with timestamp
        message_timestamp = int(time.time())
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "multi_model": use_multi_model,
            "timestamp": message_timestamp
        })
        
        if use_multi_model:
            display_multi_model_message("user", user_input)
            
            # Generate multi-model responses
            multi_results = asyncio.run(process_multi_query(user_input, top_k))
            
            # Add and display multi-model messages with timestamp
            assistant_timestamp = int(time.time())
            st.session_state.messages.append({
                "role": "assistant",
                "content": multi_results,
                "multi_model": True,
                "timestamp": assistant_timestamp
            })
            display_multi_model_message("assistant", multi_results, assistant_timestamp)
        else:
            display_message("user", user_input)
            
            # Generate single model response
            answer, sources, audio_bytes = process_query(user_input, top_k)
            
            # Add and display assistant message with preloaded audio and timestamp
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "audio_bytes": audio_bytes,
                "multi_model": False,
                "timestamp": int(time.time())
            })
            display_message("assistant", answer, sources, audio_bytes)
    
    # Footer
    st.divider()
    st.caption(
        "Powered by Gemini • ChromaDB • Sentence Transformers"
    )


if __name__ == "__main__":
    main()
