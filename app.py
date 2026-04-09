import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Load Data ---
def load_and_prep():
    df = pd.read_csv("movie_data.csv")
    df['description'] = df['description'].fillna('No description available')
    return df

df = load_and_prep()

# --- 2. Load Multilingual Model ---
print("Loading Multilingual Transformer model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Computing Vector Space Embeddings...")
movie_embeddings = model.encode(df['description'].tolist(), show_progress_bar=True)

# --- 3. Analysis Logic ---
def analyze_semantics(query_title):
    if query_title not in df['title'].values:
        return None, f"⚠️ '{query_title}' not found in dataset. Try 'Twin Peaks: Fire Walk with Me' or check spelling."
    
    idx = df[df['title'] == query_title].index[0]
    sim_scores = cosine_similarity([movie_embeddings[idx]], movie_embeddings)[0]
    
    temp_df = df.copy()
    temp_df['score'] = sim_scores
    recs = temp_df.sort_values(by='score', ascending=False).iloc[1:6]
    recs['confidence'] = recs['score'].apply(lambda x: f"{x:.2%}")
    
    return recs[['title', 'release_date', 'confidence']], "Success: Semantic neighborhood identified."

# --- 4. UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 SFF Semantic Analysis Laboratory")
    gr.Markdown("*Computational Modeling for Film Festival Scheduling & Recommendation*")
    
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(label="Enter Movie Title (English)", placeholder="e.g. Blue Velvet")
            run_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            status = gr.Label(label="Status")
            output_table = gr.DataFrame(label="Top 5 Similar Matches")

    run_btn.click(analyze_semantics, inputs=input_box, outputs=[output_table, status])

demo.launch()