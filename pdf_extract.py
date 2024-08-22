from flask import Flask, render_template, request
from fileinput import filename
from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import retrieval_qa
import os
import pymupdf, pathlib
import pytesseract

app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/chat', methods = ['POST'])   
def success():   
    if request.method == 'POST':   
        f = request.files['file']
        if not os.path.exists('files_uploaded'):
            os.makedirs('files_uploaded')
        f.save(f"files_uploaded/{f.filename}")

        fName, fExt = f.filename.split('.')
        
        # Extract plain text
        with pymupdf.open(f"files_uploaded/{f.filename}") as doc:
            text = chr(12).join([page.get_text() for page in doc])
        pathlib.Path(f'{fName}.txt').write_bytes(text.encode())

        #Extract tables as json
        doc = pymupdf.open(f"files_uploaded/{f.filename}")
        txt_file = open(f'{fName}.txt', 'a', encoding='utf-8', errors='ignore')
        for page in doc:
            tabs = page.find_tables()
            for tab in tabs:
                df = tab.to_pandas()
                txt_file.write(df.to_json(orient='records', lines=True))

        #Extract images and text boxes from pdf
        doc = pymupdf.open(f"files_uploaded/{f.filename}")
        if not os.path.exists('images_extracted'):
            os.makedirs('images_extracted')
        for page in doc:
            new_rects = detect_rects(page)
            for i, r in enumerate(new_rects):
                pix = page.get_pixmap(dpi=150, clip=r)
                pix.save("images_extracted/images_graphic-%03i-%02i.png" % (page.number, i))

        #Extract data from extracted images and text boxes
        img_folder = 'images_extracted/'  # example: image folder name provided
        dirname = os.path.dirname(img_folder)
        img_list = os.listdir(img_folder)  # some list of image filenames
        for img in img_list:
            imgfile = os.path.join(dirname, img)
            text = pytesseract.image_to_string(imgfile, lang='eng')
            txt_file.write(text)
        
        index_doc(f'{fName}.txt')

        return render_template('chat.html')

def open_api_file():
    with open("api_key.txt", "r", encoding='utf-8') as file:
        return file.read()
    
def index_doc(file):
    os.environ["OPENAI_API_KEY"] = open_api_file()

    loader = TextLoader(file, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 0,
        length_function = len,
    )

    docs = text_splitter.split_documents(documents)

    embedding = OpenAIEmbeddings()
    library = FAISS.from_documents(docs, embedding)

    library.save_local("faiss_index")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):

    user_query = text

    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_chroma import Chroma
    from langchain_community.chat_message_histories import ChatMessageHistory
    # from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
    os.environ['OPENAI_API_KEY']='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'
    # print(os.environ['OPENAI_API_KEY'])
    # import getpass
    # import os

    # os.environ["OPENAI_API_KEY"] = getpass.getpass()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


    from langchain_community.vectorstores import FAISS

    db = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization = True)

    # query = "What does Frshr technology does"
    retireved_results=db.as_retriever()

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [    ("ai", contextualize_q_system_prompt),
            (MessagesPlaceholder("chat_history")),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retireved_results, contextualize_q_prompt
    )


    ### Answer question ###
    qa_system_prompt =  """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say "Apologies, I do not have the answer to this question."\
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("ai", qa_system_prompt),
            (MessagesPlaceholder("chat_history")),
           ("human", "{input}")
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    from langchain_core.messages import HumanMessage
    chat_history = []

    response = rag_chain.invoke({"input":user_query, "chat_history": chat_history})
    # response = user_input(user_query)
    chat_history.extend([HumanMessage(content=user_query), response["answer"]])

    return response['answer']

def detect_rects(page, graphics=None):
    """Detect and join rectangles of neighboring vector graphics."""
    delta = 3

    def are_neighbors(r1, r2):
        """Detect whether r1, r2 are "neighbors".

        Neighbors are defined as:
        The minimum distance between points of r1 and points of r2 is not
        larger than delta.

        This check supports empty rect-likes and thus also lines.
        """
        if (
            (
                r2.x0 - delta <= r1.x0 <= r2.x1 + delta
                or r2.x0 - delta <= r1.x1 <= r2.x1 + delta
            )
            and (
                r2.y0 - delta <= r1.y0 <= r2.y1 + delta
                or r2.y0 - delta <= r1.y1 <= r2.y1 + delta
            )
            or (
                r1.x0 - delta <= r2.x0 <= r1.x1 + delta
                or r1.x0 - delta <= r2.x1 <= r1.x1 + delta
            )
            and (
                r1.y0 - delta <= r2.y0 <= r1.y1 + delta
                or r1.y0 - delta <= r2.y1 <= r1.y1 + delta
            )
        ):
            return True
        return False

    # we exclude graphics not contained in reasonable page margins
    parea = page.rect + (-36, -36, 36, 36)

    if graphics is None:
        graphics = page.get_drawings()
    # exclude graphics not contained inside margins
    paths = [
        p
        for p in page.get_drawings()
        if parea.x0 <= p["rect"].x0 <= p["rect"].x1 <= parea.x1
        and parea.y0 <= p["rect"].y0 <= p["rect"].y1 <= parea.y1
    ]

    # list of all vector graphic rectangles
    prects = sorted([p["rect"] for p in paths], key=lambda r: (r.y1, r.x0))

    new_rects = []  # the final list of the joined rectangles

    # -------------------------------------------------------------------------
    # The strategy is to identify and join all rects that are neighbors
    # -------------------------------------------------------------------------
    while prects:  # the algorithm will empty this list
        r = prects[0]  # first rectangle
        repeat = True
        while repeat:
            repeat = False
            for i in range(len(prects) - 1, 0, -1):  # back to front
                if are_neighbors(prects[i], r):
                    r |= prects[i].tl  # join in to first rect
                    r |= prects[i].br
                    del prects[i]  # delete this rect
                    repeat = True

        prects[0] = +r
        # move first item over to result list
        new_rects.append(prects.pop(0))
        prects = sorted(list(set(prects)), key=lambda r: (r.y1, r.x0))

    new_rects = sorted(list(set(new_rects)), key=lambda r: (r.y1, r.x0))
    return [r for r in new_rects if r.width > 5 and r.height > 5]


if __name__ == "__main__":
    app.run(host='0.0.0.0', threaded=True, port='1000')

