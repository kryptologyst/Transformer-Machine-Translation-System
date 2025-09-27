"""
Modern Web UI for Transformer Machine Translation
Built with Streamlit for interactive translation experience
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import json
from modern_translator import ModernTranslationSystem, TranslationResult

# Page configuration
st.set_page_config(
    page_title="🌐 Modern Transformer Translation",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .translation-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'translator' not in st.session_state:
    st.session_state.translator = ModernTranslationSystem()
if 'translation_history' not in st.session_state:
    st.session_state.translation_history = []

def main():
    # Header
    st.markdown('<h1 class="main-header">🌐 Modern Transformer Translation</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Select Translation Model",
            ["marianmt", "t5", "mbart", "opus"],
            help="Choose the transformer model for translation"
        )
        
        # Language selection
        col1, col2 = st.columns(2)
        with col1:
            source_lang = st.selectbox(
                "Source Language",
                ["en", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko"],
                index=0
            )
        
        with col2:
            target_lang = st.selectbox(
                "Target Language",
                ["en", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko"],
                index=1
            )
        
        # Advanced options
        st.header("🔧 Advanced Options")
        max_length = st.slider("Max Translation Length", 50, 512, 256)
        num_beams = st.slider("Number of Beams", 1, 10, 4)
        
        # Sample texts
        st.header("📝 Sample Texts")
        sample_texts = [
            "Hello, how are you?",
            "The weather is beautiful today.",
            "I love artificial intelligence.",
            "Machine learning is fascinating.",
            "Natural language processing is amazing."
        ]
        
        selected_sample = st.selectbox("Choose a sample text", ["Custom"] + sample_texts)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Translation Interface")
        
        # Text input
        if selected_sample == "Custom":
            text_input = st.text_area(
                "Enter text to translate:",
                height=150,
                placeholder="Type your text here..."
            )
        else:
            text_input = st.text_area(
                "Enter text to translate:",
                value=selected_sample,
                height=150
            )
        
        # Batch translation
        st.subheader("📄 Batch Translation")
        batch_text = st.text_area(
            "Enter multiple texts (one per line):",
            height=100,
            placeholder="Enter multiple texts, one per line..."
        )
        
        # Translation buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            translate_single = st.button("🔄 Translate Single Text", type="primary")
        
        with col_btn2:
            translate_batch = st.button("📄 Translate Batch", type="secondary")
    
    with col2:
        st.header("📊 Statistics")
        
        # Model info
        st.subheader("Model Information")
        st.info(f"**Selected Model:** {model_type.upper()}")
        st.info(f"**Language Pair:** {source_lang.upper()} → {target_lang.upper()}")
        
        # Performance metrics
        if st.session_state.translation_history:
            avg_time = sum(t.processing_time for t in st.session_state.translation_history) / len(st.session_state.translation_history)
            st.metric("Average Processing Time", f"{avg_time:.3f}s")
            st.metric("Total Translations", len(st.session_state.translation_history))
    
    # Translation results
    if translate_single and text_input.strip():
        with st.spinner("Translating..."):
            try:
                result = st.session_state.translator.translate(
                    text_input, source_lang, target_lang, model_type
                )
                
                st.session_state.translation_history.append(result)
                
                # Display results
                st.success("✅ Translation completed!")
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.markdown('<div class="translation-box">', unsafe_allow_html=True)
                    st.subheader(f"📝 Original ({source_lang.upper()})")
                    st.write(result.original_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_res2:
                    st.markdown('<div class="translation-box">', unsafe_allow_html=True)
                    st.subheader(f"🔄 Translated ({target_lang.upper()})")
                    st.write(result.translated_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Metrics
                col_met1, col_met2, col_met3 = st.columns(3)
                with col_met1:
                    st.metric("Processing Time", f"{result.processing_time:.3f}s")
                with col_met2:
                    st.metric("Model", result.model_name.split('/')[-1])
                with col_met3:
                    st.metric("Characters", len(result.translated_text))
                
            except Exception as e:
                st.error(f"❌ Translation failed: {str(e)}")
    
    # Batch translation results
    if translate_batch and batch_text.strip():
        texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
        
        if texts:
            with st.spinner(f"Translating {len(texts)} texts..."):
                try:
                    results = st.session_state.translator.batch_translate(
                        texts, source_lang, target_lang, model_type
                    )
                    
                    st.session_state.translation_history.extend(results)
                    
                    st.success(f"✅ Batch translation completed! ({len(texts)} texts)")
                    
                    # Display batch results
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Translation {i}: {result.original_text[:50]}..."):
                            col_b1, col_b2 = st.columns(2)
                            
                            with col_b1:
                                st.write("**Original:**")
                                st.write(result.original_text)
                            
                            with col_b2:
                                st.write("**Translated:**")
                                st.write(result.translated_text)
                            
                            st.write(f"**Processing time:** {result.processing_time:.3f}s")
                
                except Exception as e:
                    st.error(f"❌ Batch translation failed: {str(e)}")
    
    # Translation history and analytics
    if st.session_state.translation_history:
        st.header("📈 Translation Analytics")
        
        # Create DataFrame for analytics
        df = pd.DataFrame([
            {
                'timestamp': datetime.now(),
                'source_lang': t.source_lang,
                'target_lang': t.target_lang,
                'model': t.model_name.split('/')[-1],
                'processing_time': t.processing_time,
                'text_length': len(t.original_text),
                'translation_length': len(t.translated_text)
            }
            for t in st.session_state.translation_history
        ])
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Processing Time Distribution")
            fig_time = px.histogram(df, x='processing_time', nbins=20, title="Processing Time Distribution")
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col_chart2:
            st.subheader("Model Usage")
            model_counts = df['model'].value_counts()
            fig_model = px.pie(values=model_counts.values, names=model_counts.index, title="Model Usage")
            st.plotly_chart(fig_model, use_container_width=True)
        
        # Language pair analysis
        st.subheader("Language Pair Analysis")
        lang_pairs = df.groupby(['source_lang', 'target_lang']).size().reset_index(name='count')
        lang_pairs['pair'] = lang_pairs['source_lang'] + ' → ' + lang_pairs['target_lang']
        
        fig_lang = px.bar(lang_pairs, x='pair', y='count', title="Translation Language Pairs")
        st.plotly_chart(fig_lang, use_container_width=True)
        
        # Recent translations table
        st.subheader("Recent Translations")
        recent_df = df.tail(10)[['source_lang', 'target_lang', 'model', 'processing_time', 'text_length']]
        st.dataframe(recent_df, use_container_width=True)
    
    # Evaluation section
    st.header("🎯 Translation Quality Evaluation")
    
    # Get sample data for evaluation
    sample_data = st.session_state.translator.database.get_sample_translations(source_lang, target_lang)
    
    if sample_data:
        st.subheader("Sample Translation Evaluation")
        
        selected_sample = st.selectbox(
            "Select sample for evaluation",
            [f"{s['original']} → {s['reference']}" for s in sample_data]
        )
        
        if selected_sample:
            sample_idx = [f"{s['original']} → {s['reference']}" for s in sample_data].index(selected_sample)
            sample = sample_data[sample_idx]
            
            if st.button("🔍 Evaluate Translation Quality"):
                with st.spinner("Evaluating translation quality..."):
                    try:
                        # Translate the sample
                        result = st.session_state.translator.translate(
                            sample['original'], source_lang, target_lang, model_type
                        )
                        
                        # Evaluate quality
                        metrics = st.session_state.translator.evaluate_translation(
                            sample['original'], result.translated_text, sample['reference']
                        )
                        
                        # Display results
                        col_eval1, col_eval2 = st.columns(2)
                        
                        with col_eval1:
                            st.write("**Original:**", sample['original'])
                            st.write("**Reference:**", sample['reference'])
                            st.write("**Translation:**", result.translated_text)
                        
                        with col_eval2:
                            st.metric("BLEU Score", f"{metrics['bleu']:.2f}")
                            st.metric("ROUGE-1", f"{metrics['rouge1']:.2f}")
                            st.metric("ROUGE-2", f"{metrics['rouge2']:.2f}")
                            st.metric("ROUGE-L", f"{metrics['rougeL']:.2f}")
                            if metrics['bert_f1'] > 0:
                                st.metric("BERT F1", f"{metrics['bert_f1']:.2f}")
                    
                    except Exception as e:
                        st.error(f"❌ Evaluation failed: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🌐 **Modern Transformer Translation System** | "
        "Built with Streamlit, Transformers, and PyTorch | "
        "Supporting MarianMT, T5, mBART, and OPUS models"
    )

if __name__ == "__main__":
    main()
