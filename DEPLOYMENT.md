# 🚀 Deployment Guide

This guide covers various deployment options for the Modern Transformer Machine Translation System.

## 📋 Prerequisites

- Python 3.8+ or Docker
- Internet connection for model downloads
- Sufficient disk space (5-10GB for models)
- Memory: 4GB+ RAM recommended

## 🐳 Docker Deployment (Recommended)

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd transformer-machine-translation

# Build and run with Docker Compose
docker-compose up -d

# Access the application
open http://localhost:8501
```

### Manual Docker Build
```bash
# Build the image
docker build -t transformer-translation .

# Run the container
docker run -p 8501:8501 -v $(pwd)/data:/app/data transformer-translation
```

## 🐍 Python Deployment

### Local Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run basic demo
python 0126.py

# Run advanced system
python modern_translator.py

# Launch web interface
streamlit run streamlit_app.py
```

### Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run streamlit_app.py
```

## ☁️ Cloud Deployment

### Heroku
1. Create `Procfile`:
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Deploy:
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### AWS EC2
1. Launch EC2 instance (t3.medium or larger)
2. Install Docker:
   ```bash
   sudo yum update -y
   sudo yum install -y docker
   sudo service docker start
   sudo usermod -a -G docker ec2-user
   ```

3. Deploy application:
   ```bash
   git clone <repository-url>
   cd transformer-machine-translation
   docker-compose up -d
   ```

### Google Cloud Platform
1. Create Compute Engine instance
2. Install Docker and deploy:
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   git clone <repository-url>
   cd transformer-machine-translation
   docker-compose up -d
   ```

### Azure Container Instances
1. Build and push to Azure Container Registry
2. Deploy using Azure CLI:
   ```bash
   az container create \
     --resource-group myResourceGroup \
     --name transformer-translation \
     --image myregistry.azurecr.io/transformer-translation:latest \
     --ports 8501 \
     --dns-name-label transformer-translation
   ```

## 🔧 Configuration

### Environment Variables
```bash
# Model configuration
MODEL_CACHE_DIR=/app/models
MAX_MODEL_SIZE=5GB

# Database configuration
DATABASE_URL=sqlite:///app/data/translations.db

# Performance settings
BATCH_SIZE=8
MAX_CONCURRENT_TRANSLATIONS=4
ENABLE_GPU=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/translation.log
```

### Production Settings
For production deployment, consider:

1. **Database**: Use PostgreSQL instead of SQLite
2. **Caching**: Implement Redis for model caching
3. **Load Balancing**: Use nginx or similar
4. **Monitoring**: Add health checks and metrics
5. **Security**: Implement authentication and rate limiting

## 📊 Performance Optimization

### Model Optimization
```python
# Enable model quantization
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Use half precision
model = model.half()

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

### Memory Optimization
```python
# Clear cache periodically
torch.cuda.empty_cache()

# Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = outputs.loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 🔍 Monitoring and Logging

### Health Checks
```python
# Add to your application
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(translator.models),
        'memory_usage': psutil.virtual_memory().percent
    }
```

### Logging Configuration
```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('translation.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
```

## 🛡️ Security Considerations

1. **Input Validation**: Sanitize all user inputs
2. **Rate Limiting**: Implement request rate limiting
3. **Authentication**: Add user authentication if needed
4. **HTTPS**: Use SSL/TLS in production
5. **Model Security**: Validate model integrity

## 📈 Scaling Considerations

### Horizontal Scaling
- Use load balancers
- Implement session management
- Consider microservices architecture

### Vertical Scaling
- Increase instance size
- Optimize model loading
- Implement model caching

### Caching Strategies
- Model caching
- Translation result caching
- Database query caching

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce batch size
   - Use smaller models
   - Enable gradient checkpointing

2. **Slow Model Loading**:
   - Pre-download models
   - Use model caching
   - Consider model quantization

3. **Translation Quality Issues**:
   - Try different models
   - Adjust beam search parameters
   - Use domain-specific models

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export PYTHONPATH=/app
python -m streamlit run streamlit_app.py --logger.level=debug
```

## 📞 Support

For deployment issues:
- Check logs: `docker logs <container-name>`
- Monitor resources: `docker stats`
- Test connectivity: `curl http://localhost:8501/_stcore/health`

---

**Happy Deploying! 🚀**
