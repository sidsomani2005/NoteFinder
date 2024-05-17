import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub




def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


@st.cache_data
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history'][1:]

    # chat display
    for i, message in enumerate(st.session_state.chat_history[3:]):
        cleaned_message = str(message)[:-1].replace('content=\'', '').strip()
        if i % 2 == 0:
            st.markdown(
                f'<div style="background-color: #e6f2ff; border-radius: 5px; padding: 10px; margin-bottom: 10px; color: #333;">{cleaned_message}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background-color: #cce0ff; border-radius: 5px; padding: 10px; margin-bottom: 10px; color: #333;"><strong>{cleaned_message}</strong> </div>',
                unsafe_allow_html=True
            )



# method to generate questions from the uploaded file
@st.cache_data
def get_questions(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    question_output = []
    for i, question in enumerate(st.session_state.chat_history):
        cleaned_message = str(question)[:-1].replace('content=\'', '').strip()
        if i % 2 != 0:
            if cleaned_message.endswith('?'):
                cleaned_message = cleaned_message.replace('\n', ' ')
                question_output.append(cleaned_message)
            else:
                cleaned_message = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_message)  # Remove special characters
                cleaned_message = re.sub(r'\s+', ' ', cleaned_message).strip()  # Remove extra spaces
                cleaned_message = cleaned_message.capitalize()  # Capitalize the sentence
                cleaned_message = cleaned_message.replace('\n', ' ')
                question_output.append(f"{cleaned_message}?")
    return question_output





# method to generate summary from the uploaded file
@st.cache_data
def generate_summary(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    summary_messages = []
    for i, message in enumerate(st.session_state.chat_history):
        cleaned_message = str(message)[:-1].replace('content=\'', '').strip()
        if i % 2 != 0:
            summary_messages.append(cleaned_message)
    return summary_messages


# carry on from get_summary method
def get_document_summary():
    if st.session_state.conversation:
        user_question = "Provide a summary of the uploaded file. If the file is a textbook or a book, disregard the table of contents and generate a brief overview of the primary contents of the book or chapters in the book."
        return generate_summary(user_question)
    else:
        st.warning("Please upload a document first.")




# carry on from get_questions method
def get_document_questions():
    if st.session_state.conversation:
        user_question = "Generate a list of detailed questions about the content contained within uploaded document. If the file is a textbook or a book, disregard the table of contents and generate questions based on the primary contents of the book or chapters in the book. WRITE THE QUESTIONS IN QUESTION FORMAT (e.g What does this topic mean?) for the user to practice about the uploaded document. DO NOT INCLUDE THE CHARACTERS \n WITHIN THE QUESTIONS! Always end each question with a question mark!"
        return get_questions(user_question)
    else:
        st.warning("Please upload a document first.")


def main():
    load_dotenv()
    st.set_page_config(
        page_title = "NoteFinder",
        page_icon = ":notebook:"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    col1, col2 = st.columns([1,2])
    with col1:
        st.title("NoteFinder")
    #with col2:
        #st.image("/Users/sidsomani/Desktop/NoteFinder/logovs2.png", width = 80)

    for i in range(4):
        st.text("")


    with st.sidebar:
        # st.image("/Users/sidsomani/Desktop/NoteFinder/logovs2.png", width = 290)

        for i in range(2):
            st.text("")

        openai_api_key = st.text_input('', type='password')
        st.session_state.openai_api_key = openai_api_key
        if not st.session_state.openai_api_key:
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        elif not st.session_state.openai_api_key.startswith('sk-'):
            st.error("Invalid OpenAI API key!", icon='⚠')
        else:
            st.success("Valid OpenAI API key")

        openai_api_key = st.session_state.openai_api_key
        for i in range(2):
            st.text("")

        pdf_docs = st.file_uploader("Upload your files", accept_multiple_files=True)

        for i in range(2):
            st.text("")

        if pdf_docs and st.button("Process"):
            st.text("")
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store with embedded chunks
                vectorstore = get_vectorstore(text_chunks)


                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)



    tab1, tab2, tab3 = st.tabs(["Finder", "Summary", "Questions"])
    with tab1:
        st.text("")
        user_question = st.text_input("Ask a question about your documents")
        if user_question:
            handle_userinput(user_question)

    with tab2:
        st.text("")
        summary = get_document_summary()
        if summary:
            st.session_state.summary_message = summary
            for message in st.session_state.summary_message:
                st.markdown(
                    f'<div style="background-color: #cce0ff; border-radius: 5px; padding: 10px; margin-bottom: 10px; color: #333;">{message}</div>',
                    unsafe_allow_html=True
                )

    with tab3:
        st.text("")
        questions = get_document_questions()
        if questions:
            st.session_state.question_bullet = questions
            for question in st.session_state.question_bullet[1:]:
                st.markdown(
                    f'<div style="background-color: #cce0ff; border-radius: 5px; padding: 10px; margin-bottom: 10px; color: #333;">{question}</div>',
                    unsafe_allow_html=True
                )


if __name__ == '__main__':
    main()
