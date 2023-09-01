import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, ConversationChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.prompt import PromptTemplate
from htmlTemplates import css, bot_template, user_template
from openai.error import OpenAIError

template = """
The following is a friendly conversation between a professional human consultant and an AI. 
The AI is helpful and supportive and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.

As the AI, your role is to act as a business consultant specialized in writing proposals that win projects.
You will focus on how to improve the proposal to increase the likelihood of winning the bid.
You will be provided with information about the proposal that the consultant is working on.
In addition, you will receive information about the request for proposal(RFP or Anbud) that the proposal is written as a response to.
In order to success you are required to make comparisons between the proposal and the RFP to find ways to improve the proposal.
Make references to parts of the RFP and the proposal that are related as much as you can to make it easier for the consultant to understand your suggestions.   

Relevant context from the request for proposal (RFP):
{rfp_context}

This should be compared to the relevant context from the proposal that the human consultant is working  on:
{proposal_context}

Current conversation:
{chat_history}
Human: {question}
AI Assistant:"""


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Extract the raw text for all uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = OpenAI(temperature=0.5, model="text-davinci-003")
    prompt = PromptTemplate(
        input_variables=["chat_history", "question", "rfp_context", "proposal_context"],
        template=template
    )
    #conversation_chain = ConversationChain(
    #    prompt=prompt,
    #    llm=llm,
    #    chain_type="stuff",
    #    verbose=True,
    #    retriever=vectorstore.as_retriever(),
    #    memory=ConversationBufferMemory(ai_prefix="AI Assistant", memory_key='chat_history', return_messages=True)
    #)
    conversation_chain = LLMChain(
        llm=llm,
        verbose=True,
        prompt=prompt
    )

    #conversation_chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=prompt)
    return conversation_chain


def handle_userinput(user_question):
    rfp_context = st.session_state.rfp_vectorstore.similarity_search(user_question)
    #st.write(rfp_similarity)
    proposal_context = st.session_state.rfp_vectorstore.similarity_search(user_question)
    #st.write(proposal_similarity)
    #st.write(st.session_state.chat_history)

    included_history = st.session_state.chat_history
    if len(included_history) > 6:
        included_history = st.session_state.chat_history[-4:]

    response = st.session_state.conversation({
        'chat_history': included_history,
        'question': user_question,
        "rfp_context": rfp_context,
        "proposal_context": proposal_context
        },
        return_only_outputs=False
    )

    st.session_state.chat_history.append(response["question"])
    st.session_state.chat_history.append(response["text"])

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)


def clear_submit():
    st.session_state["submit"] = False


def main():
    load_dotenv()

    st.set_page_config(page_title="Proposal evaluator", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)

    # initiatilize session state on startup as best practice
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rfp_vectorstore" not in st.session_state:
        st.session_state.rfp_vectorstore = None
    if "proposal_vectorstore" not in st.session_state:
        st.session_state.proposal_vectorstore = None

    st.header("Chat with Jarvis about how to improve your proposal")
    st.write("Start your chat by providing some context for what you are working on by uploading the RFP and your proposal in the sidebar to the left")
    with st.expander("Example prompts"):
        st.write("How well does the proposal cover what is expected in the RFP?")
        st.write("What likelihood score between 0 and 100% would you give this proposal of being accepted by the customer given how well it covers the expectations outlined in the RFP?")
        st.write("How can we improve this proposal to achieve a score above 80%")
        st.write("I am unsure of how to explain our  competence in a compelling manner. Can you help me formulate some key points we should include in the propopsal to highlight our expertise?")
    documents_are_missing = True
    if st.session_state["rfp_vectorstore"] and st.session_state["proposal_vectorstore"]:
        documents_are_missing = False
    user_question = st.text_input("Ask Jarvis a question about your proposal:",
                                  on_change=clear_submit,
                                  disabled=documents_are_missing)
    print(documents_are_missing)
    if user_question:
        with st.spinner("Asking Jarvis for a response⏳"):
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your RFP", accept_multiple_files=True, type=["pdf"],
                                    help="Only PDF files are supported currently!", on_change=clear_submit)
        proposal_pdf_docs = st.file_uploader("Upload your proposal", accept_multiple_files=True, type=["pdf"],
                                    help="Only PDF files are supported currently!", on_change=clear_submit)
        if st.button("Process"):
            with st.spinner("Indexing documents... This may take a while⏳"):
                if pdf_docs is not None:
                    for file in pdf_docs:
                        if not file.name.endswith(".pdf"):
                            raise ValueError("File type not supported!")

                try:
                    ## Parse the RFP documents
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)
                    # get text chunks
                    text_chunks = get_text_chunks(raw_text)
                    # create vectory store
                    st.session_state.rfp_vectorstore = get_vectorstore(text_chunks)

                    ## Parse the proposal documents
                    # get pdf text
                    proposal_raw_text = get_pdf_text(proposal_pdf_docs)
                    # get text chunks
                    proposal_text_chunks = get_text_chunks(proposal_raw_text)
                    # create vectory store
                    st.session_state.proposal_vectorstore = get_vectorstore(proposal_text_chunks)

                    # Create conversation chain
                    # Added conversation to session state to make it available outside sidebar context
                    st.session_state.conversation = get_conversation_chain(st.session_state.proposal_vectorstore)
                except OpenAIError as e:
                    st.error(e._message)
            st.write(":thumbsup: Files are indexed and you are ready to chat")


if __name__ == '__main__':
    main()

