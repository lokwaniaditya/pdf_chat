<h1>Ask ChatBot questions from uploaded PDF</h1>
<h2>Overview</h2>
Extract Data from a PDF in form of Q&A - based on RAG implementation - Ask questions to a chatbot which will retrieve data from the PDF uploaded - A python based application that leverages the capabilities OpenAI's GPT-3.5 LLM to answer questions based on the pdf uploaded by the user. This application uses PymuPDF to extract text, tables and images from the PDF in the matter of a few seconds<br/>
<h2>Installation</h2>
Clone the repository: <br/>

``` 
git clone https://github.com/lokwaniaditya/pdf_chat.git 
```

``` 
cd pdf_chat 
```
Install the required libraries listed in the file requirements.txt

``` 
pip install -r requirements.txt 
```

Create a file named api_key.txt in the project directory and paste your OpenAI API key into it. <br/>
<h2>Usage</h2>
To run this application: <br/>
Run the file pdf_extract.py, this will then start the Flask Development server which is hosted locally on your machine, the user then can upload the PDF and ask questions based on the content provided in the PDF.
