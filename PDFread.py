import streamlit as st

# Set page config at the very top, before any other Streamlit commands
st.set_page_config(
    page_title="PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

from dataclasses import dataclass

import inference
import pdf_extract
import retrieval


@dataclass(frozen=True)
class Outcome:
    """What happened, and what to tell the user.

    Every entry point below used to return a bare string whether it
    succeeded or failed, so the caller could not tell "here is your answer"
    from "the model would not load" and rendered both as success.

    `ok` is False only when the request could not be served. A search that
    genuinely found nothing is a successful outcome with a negative answer.
    """

    ok: bool
    message: str


SUMMARIZER_MODEL = inference.SUMMARIZER_MODEL
QA_MODEL = inference.QA_MODEL

# Wall-clock budgets, checked between batches.
SUMMARY_TIME_BUDGET = 300
QA_TIME_BUDGET = 90

# torch and transformers are imported inside the loaders below rather than at
# module scope. They cost seconds to import, and Streamlit's file watcher is
# known to trip over torch's custom class registry when it is imported eagerly.


@st.cache_resource(show_spinner=False)
def ensure_sentence_tokenizer():
    """Fetch the NLTK sentence tokenizer once per process, not once per rerun."""
    import nltk

    # NLTK 3.9 replaced the `punkt` data package with `punkt_tab`. Accept
    # either so the app works across versions.
    for package in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{package}")
            return True
        except LookupError:
            continue

    for package in ("punkt_tab", "punkt"):
        try:
            if nltk.download(package, quiet=True):
                return True
        except Exception:
            continue
    return False


@st.cache_resource(show_spinner="Loading the summarisation model...")
def load_summarizer():
    """Load T5 once per process and share it across every session."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
    return AutoTokenizer.from_pretrained(SUMMARIZER_MODEL), inference.prepare(model)


@st.cache_resource(show_spinner="Loading the question-answering model...")
def load_qa():
    """Load DistilBERT once per process and share it across every session."""
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
    return AutoTokenizer.from_pretrained(QA_MODEL), inference.prepare(model)


@st.cache_resource(show_spinner=False, max_entries=4)
def build_index(chunks):
    """Index a document's chunks once, then reuse it for every question."""
    return retrieval.ChunkIndex.build(chunks)


@st.cache_data(show_spinner=False, max_entries=4)
def extract_pdf_cached(pdf_bytes, _progress=None):
    """Extract text from a PDF, keyed on the file's content.

    Re-uploading or re-processing the same document is served from cache.
    `_progress` is underscore-prefixed so Streamlit excludes it from the key.
    """
    return pdf_extract.extract(pdf_bytes, _progress)


def clear_document_state():
    """Drop everything derived from the previously loaded document.

    Without this, loading a second PDF leaves the first one's summary and
    Q&A history on screen, where they read as results for the new document.
    """
    for key in ("summary", "conversation", "last_question", "last_answer"):
        st.session_state.pop(key, None)


def throttled_progress(bar, step=0.02, status=None, label=""):
    """Return a progress callback that only redraws every `step` of the way.

    Each `st.progress` call is a websocket round-trip, so updating on every
    page or chunk costs more than the work being reported on.
    """
    last = {"fraction": -1.0}

    def report(done, total):
        fraction = done / total if total else 1.0
        if fraction - last["fraction"] >= step or done >= total:
            last["fraction"] = fraction
            bar.progress(min(1.0, fraction))
            if status is not None:
                status.text(f"{label} {done}/{total}...")

    return report


class PDFAssistant:
    def __init__(self):
        # Models are held by the module-level `@st.cache_resource` loaders, so
        # they are shared across every session instead of being rebuilt per tab.
        self.pdf_text = ""
        self.summary = ""
        # Chunking is not free, and every question re-chunks the same document,
        # so results are memoised until a new PDF is loaded.
        self._chunk_cache = {}

    def _load_models(self):
        """Warm the shared model cache. Returns False if loading failed."""
        try:
            self.summarizer_tokenizer, self.summarizer_model = load_summarizer()
            self.qa_tokenizer, self.qa_model = load_qa()
            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False

    def read_pdf(self, pdf_file):
        """Extract text from a PDF file.

        Returns an Outcome rather than a bare string. Both outcomes used
        to be plain text, so an unreadable document was announced with
        st.success() and still unlocked the Summary and Q&A tabs.
        """
        try:
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)  # Reset file pointer after reading

            progress_bar = st.progress(0)
            result = extract_pdf_cached(pdf_bytes, throttled_progress(progress_bar))
            progress_bar.empty()

            self.pdf_text = result.text
            self.summary = ""
            self._chunk_cache.clear()

            if not self.pdf_text.strip():
                return Outcome(
                    False,
                    "No text could be extracted. The document may be a scan or "
                    "images only, which needs OCR rather than text extraction.",
                )

            return Outcome(
                True,
                f"PDF loaded successfully. Contains {result.char_count} characters "
                f"and {result.page_count} pages (read with {result.backend}).",
            )
        except Exception as e:
            return Outcome(False, f"Error reading PDF: {str(e)}")
    
    def _chunks(self, kind, tokenizer, budget_tokens):
        """Token-aware chunks for `kind`, memoised for the loaded document."""
        if kind not in self._chunk_cache:
            ensure_sentence_tokenizer()
            self._chunk_cache[kind] = inference.chunk_by_tokens(
                self.pdf_text, tokenizer, budget_tokens
            )
        return self._chunk_cache[kind]

    def generate_summary(self, on_partial=None):
        """Generate a summary of the PDF content.

        `on_partial` receives section summaries as they are produced, so the
        caller can show progress with real content during a long run.
        """
        if not self.pdf_text:
            return Outcome(False, "Please load a PDF first.")

        if not self._load_models():
            return Outcome(False, "Failed to load AI models.")

        try:
            chunks = [
                chunk
                for chunk in self._chunks(
                    "summary", self.summarizer_tokenizer, inference.SUMMARY_INPUT_TOKENS
                )
                if len(chunk) >= 100  # skip fragments too short to summarise
            ]

            if not chunks:
                return Outcome(False, "Unable to extract meaningful text from the PDF.")

            # Cap the map phase so cost and output length stop tracking document
            # length. Sampling across the document beats truncating it.
            selected = retrieval.select_representative(
                chunks, inference.SUMMARY_MAX_MAP_CHUNKS
            )
            sampled = [chunks[i] for i in selected]

            progress_bar = st.progress(0)
            status_text = st.empty()
            budget = inference.Budget(SUMMARY_TIME_BUDGET)

            summary = inference.summarize_document(
                sampled,
                self.summarizer_tokenizer,
                self.summarizer_model,
                budget=budget,
                progress=throttled_progress(
                    progress_bar, status=status_text, label="Summarising section"
                ),
                on_partial=on_partial,
            )

            progress_bar.empty()
            status_text.empty()

            if not summary:
                return Outcome(
                    False,
                    "Could not generate a summary. Try a different document "
                    "or check document quality.",
                )

            if len(sampled) < len(chunks):
                st.caption(
                    f"Summarised {len(sampled)} representative sections of "
                    f"{len(chunks)}, then condensed them into one summary."
                )

            self.summary = summary

            return Outcome(True, self.summary)
        except Exception as e:
            return Outcome(False, f"Error generating the summary: {e}")

    def answer_question(self, question):
        """Answer a question based on the PDF content."""
        if not self.pdf_text:
            return Outcome(False, "Please load a PDF first.")

        if not question.strip():
            return Outcome(False, "Please enter a valid question.")

        if not self._load_models():
            return Outcome(False, "Failed to load AI models.")

        try:
            chunks = self._chunks("qa", self.qa_tokenizer, inference.QA_INPUT_TOKENS)

            if not chunks:
                return Outcome(
                    False,
                    "Unable to extract meaningful text from the PDF to answer questions.",
                )

            # Retrieve first: the QA model only reads the handful of chunks
            # worth reading, instead of every chunk in the document.
            index = build_index(chunks)
            candidates = index.search(question, k=retrieval.TOP_K)
            if not candidates:
                # No chunk shares a single term with the question.
                return Outcome(
                    True, "That question does not match anything in the document."
                )

            progress_bar = st.progress(0)
            status_text = st.empty()
            budget = inference.Budget(QA_TIME_BUDGET)

            best = inference.answer_from_chunks(
                question,
                [chunks[i] for i in candidates],
                self.qa_tokenizer,
                self.qa_model,
                budget=budget,
                progress=throttled_progress(
                    progress_bar, status=status_text, label="Reading section"
                ),
            )

            progress_bar.empty()
            status_text.empty()

            if best is None or not best.text:
                return Outcome(
                    True, "I couldn't find an answer to that question in the document."
                )

            return Outcome(True, f"{best.text} (Confidence: {best.score:.2f})")
        except Exception as e:
            return Outcome(False, f"Error answering the question: {e}")


@st.fragment
def summary_panel():
    """The Summary tab.

    A fragment, so generating a summary reruns this panel alone instead of the
    whole script -- no re-rendering the sidebar, the CSS, or the Q&A tab.
    """
    st.header("Document Summary")

    if not st.session_state.get("file_processed"):
        st.info("Please upload and process a PDF file first using the sidebar.")
        return

    if st.button("Generate Summary"):
        with st.status("Summarising...", expanded=True) as status:
            live = st.empty()
            sections = []

            def on_partial(new_sections):
                # Show sections as they land rather than a bar with nothing
                # behind it. The reduce phase rewrites these into the final
                # summary, so they are labelled as working notes.
                sections.extend(new_sections)
                live.markdown(
                    "\n\n".join(f"- {section}" for section in sections[-4:])
                )

            result = st.session_state.assistant.generate_summary(
                on_partial=on_partial
            )
            live.empty()
            status.update(
                label="Summary ready" if result.ok else "Could not summarise",
                state="complete" if result.ok else "error",
                expanded=False,
            )

        if result.ok:
            st.session_state.summary = result.message
        else:
            st.error(result.message)

    if st.session_state.get("summary"):
        st.markdown("### Summary Output")
        st.markdown('<div class="summary-container">', unsafe_allow_html=True)
        st.write(st.session_state.summary)
        st.markdown('</div>', unsafe_allow_html=True)

        st.download_button(
            "Download Summary",
            data=st.session_state.summary,
            file_name="summary.txt",
            mime="text/plain",
        )


@st.fragment
def qa_panel():
    """The Q&A tab, as a fragment so asking a question does not rerun the page."""
    st.header("Ask Questions About Your Document")

    if not st.session_state.get("file_processed"):
        st.info("Please upload and process a PDF file first using the sidebar.")
        return

    question = st.text_input("Enter your question about the document:")

    col1, col2 = st.columns([1, 3])
    with col1:
        ask_button = st.button("Ask")
    with col2:
        st.markdown("PDF is loaded and ready for questions")

    if ask_button and question:
        with st.spinner("Searching for an answer..."):
            result = st.session_state.assistant.answer_question(question)

        if result.ok:
            st.session_state.last_question = question
            st.session_state.last_answer = result.message
            # Recorded here, where an answer is actually produced. Appending
            # during rendering meant the history depended on how often the
            # page reran. Failures are shown but not recorded as answers.
            st.session_state.setdefault("conversation", []).append(
                (question, result.message)
            )
        else:
            st.error(result.message)

    if st.session_state.get("last_answer"):
        st.markdown("### Question")
        st.markdown(
            f'<div class="qa-container"><b style="color: black;">'
            f'{st.session_state.last_question}</b></div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Answer")
        st.markdown(
            f'<div class="qa-container"><span style="color: black;">'
            f'{st.session_state.last_answer}</span></div>',
            unsafe_allow_html=True,
        )

    history = st.session_state.get("conversation", [])
    if len(history) > 1:
        with st.expander("Conversation History"):
            for i, (q, a) in enumerate(history):
                st.markdown(
                    f'<div style="color: black;"><b>Q{i+1}: {q}</b></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="color: black;">A{i+1}: {a}</div>',
                    unsafe_allow_html=True,
                )
                st.divider()


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

                # Results belong to the document they came from, so they go
                # whether or not the new one loaded.
                clear_document_state()
                st.session_state.file_processed = result.ok

                if result.ok:
                    st.success(result.message)
                else:
                    st.error(result.message)
        
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
        - BM25 retrieval to find the relevant sections
        - NLTK for text processing
        - pypdfium2 for PDF extraction, with pdfplumber as a fallback
        """)

        st.divider()
        st.markdown("""
        **Performance Tips:**
        - Smaller PDFs work faster
        - Technical documents work better than scanned or image-heavy PDFs
        - Models load once per server and are shared by every session
        - Re-processing the same file is served from cache
        - Chunks are batched through the models, and sized to fill the
          512-token window rather than a quarter of it
        - Questions are answered from the sections retrieval ranks highest,
          so cost does not grow with the length of the document
        """)
    
    # Main content area - tabs for Summary and Q&A
    tab1, tab2 = st.tabs(["📝 Summary", "❓ Question & Answer"])

    with tab1:
        summary_panel()

    with tab2:
        qa_panel()


if __name__ == "__main__":
    main()