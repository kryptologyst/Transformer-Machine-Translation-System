# Configuration file for Modern Transformer Machine Translation System

# Model Configuration
MODELS = {
    "marianmt": {
        "description": "Helsinki-NLP OPUS models",
        "supported_languages": ["en", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko"],
        "default_params": {
            "max_length": 512,
            "num_beams": 4,
            "early_stopping": True
        }
    },
    "t5": {
        "description": "Text-to-Text Transfer Transformer",
        "supported_languages": ["en", "fr", "de", "es", "it", "pt", "ru"],
        "default_params": {
            "max_length": 512,
            "num_beams": 4,
            "early_stopping": True
        }
    },
    "mbart": {
        "description": "Multilingual BART",
        "supported_languages": ["en", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"],
        "default_params": {
            "max_length": 512,
            "num_beams": 4,
            "early_stopping": True
        }
    }
}

# Database Configuration
DATABASE = {
    "path": "translations.db",
    "backup_interval": 3600,  # seconds
    "max_history": 10000  # maximum translations to keep
}

# Web Interface Configuration
WEB_INTERFACE = {
    "title": "🌐 Modern Transformer Translation",
    "theme": "light",
    "sidebar_state": "expanded",
    "page_config": {
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
}

# Evaluation Metrics Configuration
EVALUATION = {
    "metrics": ["bleu", "rouge1", "rouge2", "rougeL", "bert_f1"],
    "default_reference_lang": "en",
    "bert_score_model": "microsoft/DialoGPT-medium"
}

# Performance Configuration
PERFORMANCE = {
    "batch_size": 8,
    "max_concurrent_translations": 4,
    "cache_size": 1000,
    "enable_gpu": True
}

# Logging Configuration
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "translation.log",
    "max_size": 10485760,  # 10MB
    "backup_count": 5
}
