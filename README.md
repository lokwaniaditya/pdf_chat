<h1>Ask ChatBot questions from uploaded PDF</h1>
<h2>Overview</h2>
Extract Data from a PDF in form of Q&A - based on RAG implementation - A python based application that uses flask framework and leverages the capabilities OpenAI's GPT-4o-mini LLM to answer questions based on the pdf uploaded by the user. This application uses PymuPDF to extract text, tables and images (scans images for text) from the PDF in the matter of a few seconds, and then stores that data in a vector database<br/>
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
<h2>Contribution</h2>
Contributions to the PDF chatbot are welcome! If you would like to contribute, please fork the repository, make your changes, and submit a pull request. If you find any issues or have suggestions for improvements, please open an issue on the GitHub repository.
