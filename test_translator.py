"""
Test suite for Modern Transformer Machine Translation System
"""

import pytest
import torch
from modern_translator import ModernTranslationSystem, TranslationResult, TranslationDatabase


class TestTranslationDatabase:
    """Test cases for TranslationDatabase class"""
    
    def test_database_initialization(self):
        """Test database initialization"""
        db = TranslationDatabase(":memory:")
        assert db is not None
    
    def test_sample_data_population(self):
        """Test sample data population"""
        db = TranslationDatabase(":memory:")
        samples = db.get_sample_translations("en", "fr")
        assert len(samples) > 0
        assert all("original" in sample for sample in samples)
        assert all("reference" in sample for sample in samples)
    
    def test_translation_save_and_retrieve(self):
        """Test saving and retrieving translations"""
        db = TranslationDatabase(":memory:")
        
        result = TranslationResult(
            original_text="Hello",
            translated_text="Bonjour",
            source_lang="en",
            target_lang="fr",
            model_name="test-model",
            processing_time=1.0
        )
        
        db.save_translation(result)
        # Note: In a real test, you'd retrieve and verify the saved data


class TestModernTranslationSystem:
    """Test cases for ModernTranslationSystem class"""
    
    def test_system_initialization(self):
        """Test system initialization"""
        translator = ModernTranslationSystem()
        assert translator is not None
        assert hasattr(translator, 'models')
        assert hasattr(translator, 'database')
    
    def test_available_models(self):
        """Test available models configuration"""
        translator = ModernTranslationSystem()
        available_models = translator.get_available_languages()
        
        assert "marianmt" in available_models
        assert "t5" in available_models
        assert "mbart" in available_models
        
        # Check that each model has supported languages
        for model, languages in available_models.items():
            assert isinstance(languages, list)
            assert len(languages) > 0
    
    def test_model_loading_error_handling(self):
        """Test error handling for model loading"""
        translator = ModernTranslationSystem()
        
        # Test with invalid model type
        with pytest.raises(ValueError):
            translator.load_model("invalid_model", "en", "fr")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_availability(self):
        """Test GPU availability"""
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() > 0


class TestTranslationResult:
    """Test cases for TranslationResult dataclass"""
    
    def test_translation_result_creation(self):
        """Test TranslationResult creation"""
        result = TranslationResult(
            original_text="Hello",
            translated_text="Bonjour",
            source_lang="en",
            target_lang="fr",
            model_name="test-model"
        )
        
        assert result.original_text == "Hello"
        assert result.translated_text == "Bonjour"
        assert result.source_lang == "en"
        assert result.target_lang == "fr"
        assert result.model_name == "test-model"
        assert result.confidence_score is None
        assert result.processing_time is None
    
    def test_translation_result_with_optional_fields(self):
        """Test TranslationResult with optional fields"""
        result = TranslationResult(
            original_text="Hello",
            translated_text="Bonjour",
            source_lang="en",
            target_lang="fr",
            model_name="test-model",
            confidence_score=0.95,
            processing_time=1.5
        )
        
        assert result.confidence_score == 0.95
        assert result.processing_time == 1.5


class TestEvaluationMetrics:
    """Test cases for evaluation metrics"""
    
    def test_evaluation_metrics_structure(self):
        """Test evaluation metrics structure"""
        translator = ModernTranslationSystem()
        
        # Mock evaluation
        metrics = translator.evaluate_translation(
            "Hello world",
            "Bonjour le monde",
            "Bonjour le monde"
        )
        
        assert isinstance(metrics, dict)
        assert "bleu" in metrics
        assert "rouge1" in metrics
        assert "rouge2" in metrics
        assert "rougeL" in metrics
        
        # All metrics should be numeric
        for metric, value in metrics.items():
            assert isinstance(value, (int, float))
            assert 0 <= value <= 1  # Most metrics are normalized


def test_imports():
    """Test that all required modules can be imported"""
    try:
        import transformers
        import torch
        import streamlit
        import pandas
        import plotly
        import sacrebleu
        import rouge_score
        import bert_score
        assert True
    except ImportError as e:
        pytest.fail(f"Required module not available: {e}")


def test_torch_functionality():
    """Test basic PyTorch functionality"""
    # Test tensor creation
    x = torch.tensor([1, 2, 3])
    assert x.shape == (3,)
    
    # Test basic operations
    y = x * 2
    assert torch.equal(y, torch.tensor([2, 4, 6]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
