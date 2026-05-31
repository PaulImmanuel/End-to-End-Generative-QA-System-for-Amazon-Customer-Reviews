import gradio as gr
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# --- 1. Configuration ---
# *** NEW: Point to the T5 model directory ***
MODEL_DIR = os.path.join("./models", "my_final_T5_qa_model") 
DATA_FILE = "Reviews.csv"

# --- 2. Load Model and Data ---
print("Loading fine-tuned T5 model and tokenizer...")
try:
    device = 0 if torch.cuda.is_available() else -1
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"Using device: {device_name}")

    # Load the T5 model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    # *** NEW: Use "text2text-generation" pipeline ***
    qa_pipeline = pipeline(
        "text2text-generation", # T5 is a text-generation model
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    print("✅ Model pipeline loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    qa_pipeline = None

print(f"Loading data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE, usecols=['ProductId', 'Text'])
    df.dropna(subset=['ProductId', 'Text'], inplace=True)
    print(f"✅ Data loaded. {len(df)} reviews available.")
    top_product_ids = df['ProductId'].value_counts().head(50).index.tolist()
    print(f"Found {len(top_product_ids)} top products for the dropdown.")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    df = None
    top_product_ids = []

# --- 3. Define Core Logic Function ---
def answer_question(product_id, question):
    if qa_pipeline is None or df is None:
        return "Error: Model or Data not loaded. Check console."
    
    if not product_id or not question:
        return "Please select a Product ID and enter a Question."
    
    try:
        product_reviews = df[df['ProductId'] == product_id]['Text']
        if product_reviews.empty:
            return f"No reviews found for Product ID: {product_id}."
        
        context = " \n\n--- REVIEW --- \n\n ".join(product_reviews.tolist())
        context = context[:4000] # Give T5 a bit more context
        
        # *** NEW: Format the input for T5 ***
        prompt = f"question: {question} context: {context}"
        
        print(f"\nQuerying for Product: {product_id}")
        print(f"Question: {question}")
        
        # Use the pipeline to *generate* an answer
        result = qa_pipeline(prompt, max_length=50, num_beams=4, early_stopping=True)
        
        answer = result[0]['generated_text']
        print(f"Generated answer: {answer}")
        
        return answer # Return the generated text directly
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"An error occurred: {e}"

# --- 4. Create and Launch the Gradio GUI ---
print("Building Gradio interface...")
with gr.Blocks(title="ProductQA Bot") as demo:
    gr.Markdown("# 🤖 ProductQA: AI Product Question-Answering (T5 Model)")
    gr.Markdown("Select a Product ID from the dropdown and ask a question based on its reviews.")
    
    with gr.Row():
        product_id_input = gr.Dropdown(
            label="Select a Product ID (Top 50 Most Reviewed)", 
            choices=top_product_ids
        )
        question_input = gr.Textbox(
            label="Your Question", 
            placeholder="e.g., What is the flavor like?"
        )
    
    submit_button = gr.Button("Get Answer")
    answer_output = gr.Textbox(label="Generated Answer", interactive=False)
    
    submit_button.click(
        fn=answer_question,
        inputs=[product_id_input, question_input],
        outputs=answer_output
    )

print("\n🚀 Launching GUI... Open the URL below in your browser.")
demo.launch()