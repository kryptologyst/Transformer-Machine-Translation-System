"""
Modern Transformer Machine Translation System
Supports multiple models: MarianMT, T5, mBART, and more
"""

import torch
import json
import sqlite3
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from transformers import (
    MarianTokenizer, MarianMTModel,
    T5Tokenizer, T5ForConditionalGeneration,
    MBartTokenizer, MBartForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    pipeline
)
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np


@dataclass
class TranslationResult:
    """Data class for translation results"""
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    model_name: str
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None


class TranslationDatabase:
    """Mock database for storing translation history and sample data"""
    
    def __init__(self, db_path: str = "translations.db"):
        self.db_path = db_path
        self.init_database()
        self.populate_sample_data()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Translation history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS translation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                model_name TEXT NOT NULL,
                confidence_score REAL,
                processing_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sample translations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sample_translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                original_text TEXT NOT NULL,
                reference_translation TEXT NOT NULL,
                category TEXT DEFAULT 'general'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def populate_sample_data(self):
        """Populate database with sample translation data"""
        sample_data = [
            ("en", "fr", "Hello, how are you?", "Bonjour, comment allez-vous?", "greeting"),
            ("en", "fr", "The weather is beautiful today.", "Le temps est magnifique aujourd'hui.", "weather"),
            ("en", "fr", "I love artificial intelligence.", "J'adore l'intelligence artificielle.", "technology"),
            ("en", "de", "Good morning!", "Guten Morgen!", "greeting"),
            ("en", "de", "Machine learning is fascinating.", "Maschinelles Lernen ist faszinierend.", "technology"),
            ("en", "es", "Thank you very much.", "Muchas gracias.", "greeting"),
            ("en", "es", "The future of AI is promising.", "El futuro de la IA es prometedor.", "technology"),
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for source_lang, target_lang, original, reference, category in sample_data:
            cursor.execute('''
                INSERT OR IGNORE INTO sample_translations 
                (source_lang, target_lang, original_text, reference_translation, category)
                VALUES (?, ?, ?, ?, ?)
            ''', (source_lang, target_lang, original, reference, category))
        
        conn.commit()
        conn.close()
    
    def save_translation(self, result: TranslationResult):
        """Save translation result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO translation_history 
            (original_text, translated_text, source_lang, target_lang, 
             model_name, confidence_score, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (result.original_text, result.translated_text, result.source_lang,
              result.target_lang, result.model_name, result.confidence_score,
              result.processing_time))
        
        conn.commit()
        conn.close()
    
    def get_sample_translations(self, source_lang: str, target_lang: str) -> List[Dict]:
        """Get sample translations for evaluation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT original_text, reference_translation, category
            FROM sample_translations
            WHERE source_lang = ? AND target_lang = ?
        ''', (source_lang, target_lang))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'original': row[0],
                'reference': row[1],
                'category': row[2]
            })
        
        conn.close()
        return results


class ModernTranslationSystem:
    """Modern translation system supporting multiple models and evaluation metrics"""
    
    def __init__(self):
        self.models = {}
        self.database = TranslationDatabase()
        self.available_models = {
            'marianmt': self._load_marianmt,
            't5': self._load_t5,
            'mbart': self._load_mbart,
            'opus': self._load_opus
        }
    
    def _load_marianmt(self, source_lang: str, target_lang: str):
        """Load MarianMT model"""
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            return tokenizer, model, model_name
        except Exception as e:
            print(f"Failed to load MarianMT model: {e}")
            return None, None, None
    
    def _load_t5(self, source_lang: str, target_lang: str):
        """Load T5 model"""
        model_name = "t5-small"
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            return tokenizer, model, model_name
        except Exception as e:
            print(f"Failed to load T5 model: {e}")
            return None, None, None
    
    def _load_mbart(self, source_lang: str, target_lang: str):
        """Load mBART model"""
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        try:
            tokenizer = MBartTokenizer.from_pretrained(model_name)
            model = MBartForConditionalGeneration.from_pretrained(model_name)
            return tokenizer, model, model_name
        except Exception as e:
            print(f"Failed to load mBART model: {e}")
            return None, None, None
    
    def _load_opus(self, source_lang: str, target_lang: str):
        """Load OPUS model"""
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            return tokenizer, model, model_name
        except Exception as e:
            print(f"Failed to load OPUS model: {e}")
            return None, None, None
    
    def load_model(self, model_type: str, source_lang: str, target_lang: str):
        """Load specified model"""
        if model_type not in self.available_models:
            raise ValueError(f"Model type {model_type} not supported")
        
        tokenizer, model, model_name = self.available_models[model_type](source_lang, target_lang)
        
        if tokenizer is None or model is None:
            raise RuntimeError(f"Failed to load {model_type} model")
        
        self.models[model_type] = {
            'tokenizer': tokenizer,
            'model': model,
            'name': model_name
        }
    
    def translate(self, text: str, source_lang: str, target_lang: str, 
                  model_type: str = 'marianmt') -> TranslationResult:
        """Translate text using specified model"""
        import time
        
        if model_type not in self.models:
            self.load_model(model_type, source_lang, target_lang)
        
        model_info = self.models[model_type]
        tokenizer = model_info['tokenizer']
        model = model_info['model']
        
        start_time = time.time()
        
        # Prepare input based on model type
        if model_type == 't5':
            input_text = f"translate {source_lang} to {target_lang}: {text}"
        elif model_type == 'mbart':
            tokenizer.src_lang = source_lang
            input_text = text
        else:
            input_text = text
        
        # Tokenize and translate
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
        
        # Decode output
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        processing_time = time.time() - start_time
        
        result = TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang,
            model_name=model_info['name'],
            processing_time=processing_time
        )
        
        # Save to database
        self.database.save_translation(result)
        
        return result
    
    def batch_translate(self, texts: List[str], source_lang: str, target_lang: str,
                       model_type: str = 'marianmt') -> List[TranslationResult]:
        """Translate multiple texts"""
        results = []
        for text in texts:
            result = self.translate(text, source_lang, target_lang, model_type)
            results.append(result)
        return results
    
    def evaluate_translation(self, original: str, translated: str, reference: str) -> Dict[str, float]:
        """Evaluate translation quality using multiple metrics"""
        metrics = {}
        
        # BLEU Score
        try:
            bleu = sacrebleu.sentence_bleu(translated, [reference])
            metrics['bleu'] = bleu.score
        except:
            metrics['bleu'] = 0.0
        
        # ROUGE Score
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(reference, translated)
            metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
            metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
            metrics['rougeL'] = rouge_scores['rougeL'].fmeasure
        except:
            metrics['rouge1'] = metrics['rouge2'] = metrics['rougeL'] = 0.0
        
        # BERT Score (optional, can be slow)
        try:
            P, R, F1 = bert_score([translated], [reference], lang=reference.split()[0] if reference else 'en')
            metrics['bert_f1'] = F1.item()
        except:
            metrics['bert_f1'] = 0.0
        
        return metrics
    
    def get_available_languages(self) -> Dict[str, List[str]]:
        """Get available language pairs for different models"""
        return {
            'marianmt': ['en', 'fr', 'de', 'es', 'it', 'pt', 'ru', 'zh', 'ja', 'ko'],
            't5': ['en', 'fr', 'de', 'es', 'it', 'pt', 'ru'],
            'mbart': ['en', 'fr', 'de', 'es', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar', 'hi']
        }


def main():
    """Main function demonstrating the modern translation system"""
    print("🚀 Modern Transformer Machine Translation System")
    print("=" * 50)
    
    # Initialize system
    translator = ModernTranslationSystem()
    
    # Test translations
    test_texts = [
        "Artificial Intelligence is transforming the world.",
        "Machine learning algorithms are becoming more sophisticated.",
        "Natural language processing enables better human-computer interaction."
    ]
    
    # Test different models
    models_to_test = ['marianmt', 't5']
    
    for model_type in models_to_test:
        print(f"\n📊 Testing {model_type.upper()} Model:")
        print("-" * 30)
        
        try:
            for text in test_texts:
                result = translator.translate(text, 'en', 'fr', model_type)
                print(f"Original: {result.original_text}")
                print(f"Translated: {result.translated_text}")
                print(f"Processing time: {result.processing_time:.3f}s")
                print()
        
        except Exception as e:
            print(f"Error with {model_type}: {e}")
    
    # Evaluate with sample data
    print("\n📈 Translation Quality Evaluation:")
    print("-" * 40)
    
    sample_data = translator.database.get_sample_translations('en', 'fr')
    for sample in sample_data[:2]:  # Test first 2 samples
        result = translator.translate(sample['original'], 'en', 'fr', 'marianmt')
        metrics = translator.evaluate_translation(
            sample['original'], result.translated_text, sample['reference']
        )
        
        print(f"Text: {sample['original']}")
        print(f"Translation: {result.translated_text}")
        print(f"Reference: {sample['reference']}")
        print(f"BLEU Score: {metrics['bleu']:.2f}")
        print(f"ROUGE-1: {metrics['rouge1']:.2f}")
        print()


if __name__ == "__main__":
    main()
