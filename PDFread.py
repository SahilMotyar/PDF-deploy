import streamlit as st
import torch

# Set page config at the very top, before any other Streamlit commands
st.set_page_config(
    page_title="PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simplified imports - avoid incompatible libraries
import pdfplumber
import io
import time
import base64
import nltk
from threading import Timer

# Use a simplified approach to avoid transformers pipeline issues
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForQuestionAnswering

# Download NLTK punkt package
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import sent_tokenize

class TimeoutException(Exception):
    pass

def timeout_handler():
    raise TimeoutException("Operation timed out")

class PDFAssistant:
    def __init__(self):
        # Initialize models - using a more compatibility-focused approach
        with st.spinner("Loading AI models..."):
            try:
                # Load summarization model
                self.summarizer_model = None
                self.summarizer_tokenizer = None
                self.qa_model = None
                self.qa_tokenizer = None
                
                # We'll load models only when needed to save memory
                self.models_loaded = False
            except Exception as e:
                st.error(f"Error initializing models: {str(e)}")
        
        self.pdf_text = ""
        self.summary = ""
    
    def _load_models(self):
        """Load AI models when needed"""
        if not self.models_loaded:
            try:
                # Load smaller, faster models for Streamlit compatibility
                with st.spinner("Loading AI models (first use)..."):
                    # For summarization - use T5-small instead of BART (more compatible)
                    self.summarizer_tokenizer = AutoTokenizer.from_pretrained("t5-small")
                    self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
                    
                    # For QA - use a smaller model
                    self.qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
                    self.qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
                
                self.models_loaded = True
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                return False
        return True
        
    def read_pdf(self, pdf_file):
        """Extract text from a PDF file using pdfplumber."""
        try:
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)  # Reset file pointer after reading
            
            self.pdf_text = ""
            
            # Use pdfplumber to read the PDF
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                total_pages = len(pdf.pages)
                progress_bar = st.progress(0)
                
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    self.pdf_text += page_text + "\n"
                    progress_bar.progress((page_num + 1) / total_pages)
            
            return f"PDF loaded successfully. Contains {len(self.pdf_text)} characters and {total_pages} pages."
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def _summarize_text(self, text):
        """Summarize a chunk of text using the model."""
        if not self._load_models():
            return "Failed to load AI models."
            
        # Prepare the input for T5 (expects "summarize: " prefix)
        inputs = self.summarizer_tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate summary
        summary_ids = self.summarizer_model.generate(
            inputs.input_ids, 
            max_length=100, 
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode the summary
        summary = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def generate_summary(self):
        """Generate a summary of the PDF content."""
        if not self.pdf_text:
            return "Please load a PDF first."
        
        try:
            # Break the text into manageable chunks
            chunks = self._split_text(self.pdf_text, max_length=500, overlap=50)
            
            if not chunks:
                return "Unable to extract meaningful text from the PDF."
            
            summaries = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, chunk in enumerate(chunks):
                progress_value = (i + 1) / len(chunks)
                progress_bar.progress(progress_value)
                status_text.text(f"Processing chunk {i+1}/{len(chunks)}...")
                
                if len(chunk) < 100:  # Skip very short chunks
                    continue
                    
                # Generate summary for this chunk with timeout
                try:
                    timeout_flag = False
                    timer = Timer(60, timeout_handler)  # 60 second timeout
                    timer.start()
                    try:
                        chunk_summary = self._summarize_text(chunk)
                        summaries.append(chunk_summary)
                    except TimeoutException:
                        timeout_flag = True
                        st.warning(f"Chunk {i+1} took too long to summarize. Skipping.")
                    finally:
                        timer.cancel()
                    
                    if timeout_flag:
                        continue
                        
                except Exception as e:
                    st.error(f"Error summarizing chunk {i+1}: {str(e)}")
            
            # Combine the summaries
            if not summaries:
                return "Could not generate a summary. Try a different document or check document quality."
                
            self.summary = " ".join(summaries)
            
            return self.summary
        except Exception as e:
            st.error(f"Error in summary generation: {str(e)}")
            return "An error occurred while generating the summary."
    
    def _answer_question_from_context(self, question, context):
        """Answer a question based on the given context."""
        if not self._load_models():
            return "Failed to load AI models.", 0

        # Tokenize the input
        inputs = self.qa_tokenizer(question, context, return_tensors="pt", 
                                  truncation=True, max_length=512,
                                  padding="max_length")
        
        # Get the answer
        with torch.no_grad():
            outputs = self.qa_model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = self.qa_tokenizer.convert_tokens_to_string(
                self.qa_tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
            )
        
        # Calculate confidence score (simplified)
        confidence = float(torch.max(outputs.start_logits).item() + torch.max(outputs.end_logits).item()) / 2
        normalized_conf = min(1.0, max(0.0, confidence / 10.0))  # Normalize to 0-1
        
        return answer, normalized_conf
    
    def answer_question(self, question):
        """Answer a question based on the PDF content."""
        if not self.pdf_text:
            return "Please load a PDF first."
        
        if not question.strip():
            return "Please enter a valid question."
        
        try:
            # Add torch import here to avoid issues with module not available at top level
            import torch
            
            # For long documents, find the most relevant sections
            chunks = self._split_text(self.pdf_text, max_length=1000, overlap=100)
            
            if not chunks:
                return "Unable to extract meaningful text from the PDF to answer questions."
            
            best_answer = ""
            highest_score = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, chunk in enumerate(chunks):
                progress_value = (i + 1) / len(chunks)
                progress_bar.progress(progress_value)
                status_text.text(f"Searching chunk {i+1}/{len(chunks)}...")
                
                try:
                    timeout_flag = False
                    timer = Timer(30, timeout_handler)  # 30 second timeout per chunk
                    timer.start()
                    try:
                        answer, score = self._answer_question_from_context(question, chunk)
                        
                        if score > highest_score and len(answer.strip()) > 0:
                            highest_score = score
                            best_answer = answer
                    except TimeoutException:
                        timeout_flag = True
                        st.warning(f"Chunk {i+1} took too long to process. Skipping.")
                    finally:
                        timer.cancel()
                    
                    if timeout_flag:
                        continue
                        
                except Exception as e:
                    st.error(f"Error processing chunk {i+1}: {str(e)}")
            
            progress_bar.empty()
            status_text.empty()
            
            if not best_answer:
                return "I couldn't find an answer to that question in the document."
                
            return f"{best_answer} (Confidence: {highest_score:.2f})"
        except Exception as e:
            st.error(f"Error in question answering: {str(e)}")
            return "An error occurred while processing your question."
    
    def _split_text(self, text, max_length=1000, overlap=100):
        """Split text into overlapping chunks of approximately max_length characters."""
        if not text or text.isspace():
            return []
            
        # First split by sentences to avoid cutting in the middle of a sentence
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            st.error(f"Error tokenizing text: {str(e)}")
            # Fallback to simple splitting if tokenization fails
            sentences = [s + "." for s in text.split(".") if s]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence
            else:
                if current_chunk.strip():  # Only add non-empty chunks
                    chunks.append(current_chunk.strip())
                # Start a new chunk with overlap from the previous chunk
                overlap_point = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_point:] + " " + sentence
        
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks


# Function to create a download link for text
def get_download_link(text, filename, link_text):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


def main():
    # Add custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    .summary-container, .qa-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        color: black;  /* Ensure text is black */
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
    }
    .status-info {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    /* Ensure all text in containers is black */
    .qa-container b, .qa-container p, .summary-container p {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize the PDF Assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = PDFAssistant()
    
    # Title and description
    st.title("📄 PDF Assistant")
    st.markdown("Upload a PDF file to summarize and ask questions about it.")
    
    # Sidebar for file upload and basic info
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            if st.button("Process PDF"):
                with st.spinner("Reading PDF..."):
                    result = st.session_state.assistant.read_pdf(uploaded_file)
                    st.session_state.file_processed = True
                    st.success(result)
        
        st.markdown('<div class="status-info">', unsafe_allow_html=True)
        st.markdown("**App Status**")
        if 'file_processed' in st.session_state and st.session_state.file_processed:
            st.markdown("✅ PDF loaded")
        else:
            st.markdown("❌ No PDF loaded")
            
        if 'summary' in st.session_state and st.session_state.summary:
            st.markdown("✅ Summary generated")
        else:
            st.markdown("❌ No summary available")
        st.markdown('</div>', unsafe_allow_html=True)
                    
        st.divider()
        st.header("About")
        st.markdown("""
        This application allows you to:
        - Upload and process PDF documents
        - Generate summaries of document content
        - Ask questions about the document
        
        Powered by:
        - Hugging Face Transformers
        - T5 for summarization
        - DistilBERT for question answering
        - NLTK for text processing
        - pdfplumber for PDF extraction
        """)
        
        st.divider()
        st.markdown("""
        **Performance Tips:**
        - Smaller PDFs work faster
        - Technical documents work better than scanned or image-heavy PDFs
        - Models are loaded only when needed to improve performance
        """)
    
    # Main content area - tabs for Summary and Q&A
    tab1, tab2 = st.tabs(["📝 Summary", "❓ Question & Answer"])
    
    with tab1:
        st.header("Document Summary")
        if 'file_processed' in st.session_state and st.session_state.file_processed:
            if st.button("Generate Summary"):
                with st.spinner("Generating summary... This may take a few minutes."):
                    summary = st.session_state.assistant.generate_summary()
                    st.session_state.summary = summary
            
            if 'summary' in st.session_state and st.session_state.summary:
                st.markdown("### Summary Output")
                st.markdown('<div class="summary-container">', unsafe_allow_html=True)
                st.write(st.session_state.summary)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download button for summary
                st.markdown(
                    get_download_link(st.session_state.summary, "summary.txt", "Download Summary"),
                    unsafe_allow_html=True
                )
        else:
            st.info("Please upload and process a PDF file first using the sidebar.")
    
    with tab2:
        st.header("Ask Questions About Your Document")
        if 'file_processed' in st.session_state and st.session_state.file_processed:
            question = st.text_input("Enter your question about the document:")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                ask_button = st.button("Ask")
            with col2:
                if 'file_processed' in st.session_state and st.session_state.file_processed:
                    st.markdown("PDF is loaded and ready for questions")
            
            if ask_button and question:
                with st.spinner("Searching for an answer..."):
                    answer = st.session_state.assistant.answer_question(question)
                    st.session_state.last_answer = answer
                    st.session_state.last_question = question
            
            if 'last_answer' in st.session_state and 'last_question' in st.session_state:
                st.markdown("### Question")
                st.markdown(f'<div class="qa-container"><b style="color: black;">{st.session_state.last_question}</b></div>', unsafe_allow_html=True)
                
                st.markdown("### Answer")
                st.markdown(f'<div class="qa-container"><span style="color: black;">{st.session_state.last_answer}</span></div>', unsafe_allow_html=True)
                
                # Save conversation
                if 'conversation' not in st.session_state:
                    st.session_state.conversation = []
                
                # Add to conversation if not already added
                if not st.session_state.conversation or st.session_state.conversation[-1][0] != st.session_state.last_question:
                    st.session_state.conversation.append((st.session_state.last_question, st.session_state.last_answer))
            
            # Show conversation history
            if 'conversation' in st.session_state and len(st.session_state.conversation) > 1:
                with st.expander("Conversation History"):
                    for i, (q, a) in enumerate(st.session_state.conversation):
                        st.markdown(f'<div style="color: black;"><b>Q{i+1}: {q}</b></div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="color: black;">A{i+1}: {a}</div>', unsafe_allow_html=True)
                        st.divider()
        else:
            st.info("Please upload and process a PDF file first using the sidebar.")


if __name__ == "__main__":
    main()