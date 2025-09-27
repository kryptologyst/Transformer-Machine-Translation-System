"""
Project 126: Modern Transformer Machine Translation System
==========================================================

The Transformer model introduced by Vaswani et al. revolutionized machine translation 
by relying entirely on attention mechanisms. This modern implementation supports 
multiple transformer models and provides comprehensive evaluation metrics.

Features:
- Multiple model support (MarianMT, T5, mBART, OPUS)
- Batch translation capabilities
- Translation quality evaluation (BLEU, ROUGE, BERT Score)
- SQLite database for translation history
- Modern web UI with Streamlit
- Comprehensive error handling and logging

Usage:
    python 0126.py                    # Run basic translation demo
    python modern_translator.py       # Run advanced translation system
    streamlit run streamlit_app.py    # Launch web interface

Installation:
    pip install -r requirements.txt
"""

import torch
import time
from transformers import MarianTokenizer, MarianMTModel, pipeline
import warnings
warnings.filterwarnings("ignore")


def basic_translation_demo():
    """Basic translation demonstration using MarianMT model"""
    print("🚀 Basic Transformer Translation Demo")
    print("=" * 50)
    
    # Define source and target language
    src_lang = "en"   # English
    tgt_lang = "fr"   # French
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    
    print(f"📥 Loading model: {model_name}")
    start_time = time.time()
    
    try:
        # Load tokenizer and model
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        
        # Test sentences
        test_sentences = [
            "Artificial Intelligence is transforming the world.",
            "Machine learning algorithms are becoming more sophisticated.",
            "Natural language processing enables better human-computer interaction.",
            "Deep learning models can understand complex patterns in data.",
            "The future of AI is promising and full of possibilities."
        ]
        
        print(f"\n🔄 Translating {len(test_sentences)} sentences...")
        print("-" * 50)
        
        total_translation_time = 0
        
        for i, text in enumerate(test_sentences, 1):
            print(f"\n{i}. Original ({src_lang.upper()}): {text}")
            
            # Tokenize and translate
            translation_start = time.time()
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                translated = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
            
            # Decode output
            output = tokenizer.decode(translated[0], skip_special_tokens=True)
            translation_time = time.time() - translation_start
            total_translation_time += translation_time
            
            print(f"   Translated ({tgt_lang.upper()}): {output}")
            print(f"   ⏱️  Translation time: {translation_time:.3f}s")
        
        print(f"\n📊 Summary:")
        print(f"   Total translation time: {total_translation_time:.3f}s")
        print(f"   Average time per sentence: {total_translation_time/len(test_sentences):.3f}s")
        print(f"   Model: {model_name}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Make sure you have internet connection and sufficient disk space for model download")


def advanced_translation_demo():
    """Advanced translation using Hugging Face pipeline"""
    print("\n🔬 Advanced Translation Pipeline Demo")
    print("=" * 50)
    
    try:
        # Create translation pipeline
        print("📥 Creating translation pipeline...")
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
        
        # Test with different text types
        test_cases = [
            {
                "text": "Hello, how are you today?",
                "type": "Casual conversation"
            },
            {
                "text": "The machine learning algorithm achieved 95% accuracy on the test dataset.",
                "type": "Technical content"
            },
            {
                "text": "The weather is beautiful today, perfect for a walk in the park.",
                "type": "Descriptive text"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n{i}. {case['type']}:")
            print(f"   Original: {case['text']}")
            
            start_time = time.time()
            result = translator(case['text'])
            translation_time = time.time() - start_time
            
            print(f"   Translated: {result[0]['translation_text']}")
            print(f"   ⏱️  Time: {translation_time:.3f}s")
    
    except Exception as e:
        print(f"❌ Pipeline error: {str(e)}")


def main():
    """Main function to run all demonstrations"""
    print("🌐 Modern Transformer Machine Translation System")
    print("=" * 60)
    print("This demo showcases the power of transformer models for machine translation.")
    print("The system supports multiple models and provides comprehensive evaluation.\n")
    
    # Run basic demo
    basic_translation_demo()
    
    # Run advanced demo
    advanced_translation_demo()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Run 'python modern_translator.py' for advanced features")
    print("2. Run 'streamlit run streamlit_app.py' for web interface")
    print("3. Check 'requirements.txt' for all dependencies")
    print("4. Explore the SQLite database for translation history")
    print("\n💡 For more features, use the modern_translator.py module!")


if __name__ == "__main__":
    main()