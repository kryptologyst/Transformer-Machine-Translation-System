# Transformer Machine Translation System

A comprehensive machine translation system built with state-of-the-art transformer models, featuring multiple model support, quality evaluation metrics, and a modern web interface.

## Features

- **Multiple Model Support**: MarianMT, T5, mBART, and OPUS models
- **Batch Translation**: Process multiple texts simultaneously
- **Quality Evaluation**: BLEU, ROUGE, and BERT Score metrics
- **Interactive Web UI**: Modern Streamlit interface with real-time analytics
- **Translation History**: SQLite database for storing translation records
- **Comprehensive Error Handling**: Robust error management and logging
- **Performance Monitoring**: Processing time tracking and analytics

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- Streamlit 1.28+
- Additional dependencies listed in `requirements.txt`

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd transformer-machine-translation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python 0126.py
   ```

## Usage

### Basic Translation Demo
```bash
python 0126.py
```
Runs a basic demonstration with MarianMT model, showing translation examples and performance metrics.

### Advanced Translation System
```bash
python modern_translator.py
```
Launches the full-featured translation system with multiple models and evaluation capabilities.

### Web Interface
```bash
streamlit run streamlit_app.py
```
Opens the interactive web interface in your browser (typically at `http://localhost:8501`).

## Supported Models

| Model | Description | Languages | Use Case |
|-------|-------------|-----------|----------|
| **MarianMT** | Helsinki-NLP OPUS models | 100+ pairs | General translation |
| **T5** | Text-to-Text Transfer Transformer | 7 languages | Multitask learning |
| **mBART** | Multilingual BART | 50 languages | Multilingual tasks |
| **OPUS** | Helsinki-NLP models | 100+ pairs | Specialized domains |

## Supported Languages

- **European**: English, French, German, Spanish, Italian, Portuguese, Russian
- **Asian**: Chinese, Japanese, Korean, Hindi
- **Middle Eastern**: Arabic, Hebrew
- **And many more...**

## Evaluation Metrics

The system provides comprehensive translation quality assessment:

- **BLEU Score**: Measures n-gram precision
- **ROUGE Metrics**: ROUGE-1, ROUGE-2, ROUGE-L for recall evaluation
- **BERT Score**: Semantic similarity using BERT embeddings
- **Processing Time**: Performance monitoring

## Database Schema

The SQLite database includes two main tables:

### Translation History
```sql
CREATE TABLE translation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_text TEXT NOT NULL,
    translated_text TEXT NOT NULL,
    source_lang TEXT NOT NULL,
    target_lang TEXT NOT NULL,
    model_name TEXT NOT NULL,
    confidence_score REAL,
    processing_time REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Sample Translations
```sql
CREATE TABLE sample_translations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_lang TEXT NOT NULL,
    target_lang TEXT NOT NULL,
    original_text TEXT NOT NULL,
    reference_translation TEXT NOT NULL,
    category TEXT DEFAULT 'general'
);
```

## 🔧 Configuration

### Model Selection
Choose from available models in the web interface or programmatically:

```python
from modern_translator import ModernTranslationSystem

translator = ModernTranslationSystem()
translator.load_model('marianmt', 'en', 'fr')
result = translator.translate("Hello world!", 'en', 'fr', 'marianmt')
```

### Advanced Parameters
- **Max Length**: Control maximum translation length (50-512 tokens)
- **Number of Beams**: Beam search width (1-10)
- **Early Stopping**: Enable/disable early stopping

## Web Interface Features

### Translation Interface
- **Single Text Translation**: Real-time translation with instant results
- **Batch Translation**: Process multiple texts simultaneously
- **Sample Text Library**: Pre-loaded examples for testing
- **Model Comparison**: Side-by-side model performance

### Analytics Dashboard
- **Processing Time Distribution**: Performance analysis
- **Model Usage Statistics**: Usage patterns and preferences
- **Language Pair Analysis**: Translation frequency by language
- **Recent Translations**: Quick access to translation history

### Quality Evaluation
- **Sample Evaluation**: Test translation quality against references
- **Metric Visualization**: Interactive charts for quality metrics
- **Comparative Analysis**: Compare different models' performance

## Testing

Run the test suite to verify functionality:

```bash
# Basic functionality test
python 0126.py

# Advanced features test
python modern_translator.py

# Web interface test
streamlit run streamlit_app.py
```

## Performance Benchmarks

Typical performance on a modern CPU:

| Model | Avg. Processing Time | Memory Usage | Quality Score |
|-------|---------------------|--------------|---------------|
| MarianMT | 0.5-2.0s | 1-2GB | High |
| T5-Small | 1.0-3.0s | 2-3GB | Medium |
| mBART | 2.0-5.0s | 4-6GB | High |

*Performance varies based on text length, hardware, and model size.*

## Troubleshooting

### Common Issues

1. **Model Download Failures**:
   - Ensure stable internet connection
   - Check available disk space (models can be 1-5GB)
   - Verify Hugging Face Hub access

2. **Memory Issues**:
   - Use smaller models (T5-small vs T5-large)
   - Reduce batch size
   - Enable model quantization

3. **Translation Quality**:
   - Try different models for your language pair
   - Adjust beam search parameters
   - Use domain-specific models when available

### Error Codes

- `MODEL_LOAD_ERROR`: Model failed to load
- `TRANSLATION_ERROR`: Translation process failed
- `EVALUATION_ERROR`: Quality evaluation failed
- `DATABASE_ERROR`: Database operation failed

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Hugging Face**: For the transformers library and model hub
- **Helsinki-NLP**: For the OPUS-MT models
- **Google**: For the T5 model architecture
- **Facebook**: For the mBART model
- **Streamlit**: For the web interface framework

## References

1. Vaswani, A., et al. "Attention is all you need." NIPS 2017.
2. Lewis, M., et al. "BART: Denoising sequence-to-sequence pre-training." ACL 2020.
3. Raffel, C., et al. "Exploring the limits of transfer learning with a unified text-to-text transformer." JMLR 2020.


# Transformer-Machine-Translation-System
