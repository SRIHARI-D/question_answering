from flask import Flask, request, jsonify
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# Set your API key securely
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual API key

# Initialize embeddings
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Initialize LLM
llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama3-70b-8192"
)

# Combined method for context and question
@app.route('/ask', methods=['GET'])
def ask():
    # Get context and question from query parameters
    context = request.args.get('context')
    question = request.args.get('question')

    # Validate inputs
    if not context:
        return jsonify({"error": "Context is required"}), 400
    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Create prompt
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the provided context and the information present in LLM.
        <context>
        {context}
        </context>
        Question: {qinput}
        """
    )

    formatted_prompt = prompt.format(context=context, qinput=question)

    try:
        response = llm.invoke(formatted_prompt)
        return jsonify({"response": response.content}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
