# 🚀 Streamlit Cloud Deployment Guide

## Prerequisites

1. **GitHub Repository**: Your code must be in a public GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **OpenAI API Key**: Required for AI-powered features

## Step-by-Step Deployment

### 1. Prepare Your Repository

Ensure your repository contains:
- ✅ `streamlit_app.py` (main application file)
- ✅ `requirements.txt` (Python dependencies)
- ✅ `.streamlit/config.toml` (Streamlit configuration)
- ✅ `packages.txt` (system packages)
- ✅ All agent files in `agents/` directory
- ✅ Sample datasets in `sample_data/` directory

### 2. Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Click "New app"**
3. **Connect your GitHub account** if not already connected
4. **Select your repository**: `PrantarBorah/Multi_Agentic_System`
5. **Set the main file path**: `streamlit_app.py`
6. **Configure app settings**:
   - **App URL**: Choose a custom URL (e.g., `ml-pipeline-orchestrator`)
   - **Python version**: 3.10
   - **Branch**: main

### 3. Set Environment Variables

In the Streamlit Cloud dashboard, add these secrets:

```toml
[secrets]
OPENAI_API_KEY = "your_openai_api_key_here"
```

**Important**: Replace `your_openai_api_key_here` with your actual OpenAI API key.

### 4. Deploy and Test

1. **Click "Deploy!"**
2. **Wait for deployment** (usually 2-5 minutes)
3. **Test the application** once deployed
4. **Share the public URL** with your audience

## Post-Deployment Checklist

### ✅ Verify Core Features
- [ ] App loads without errors
- [ ] Sample datasets are accessible
- [ ] File upload functionality works
- [ ] Problem type detection works
- [ ] Data cleaning pipeline executes
- [ ] EDA visualizations render
- [ ] Model training completes
- [ ] Model evaluation works
- [ ] Dark mode toggles properly

### ✅ Test Educational Features
- [ ] Contextual learning popups appear
- [ ] ML glossary is accessible
- [ ] Decision logs are displayed
- [ ] Educational insights are shown
- [ ] Interactive tooltips work

### ✅ Performance Verification
- [ ] App loads within 30 seconds
- [ ] Pipeline execution completes in reasonable time
- [ ] Visualizations render properly
- [ ] No memory errors during execution

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Check `requirements.txt` includes all dependencies
   - Verify Python version compatibility

2. **API Key Issues**
   - Ensure OpenAI API key is correctly set in secrets
   - Check API key has sufficient credits

3. **Memory Issues**
   - Large datasets may cause memory problems
   - Consider adding memory optimization

4. **File Path Issues**
   - Ensure all file paths are relative
   - Check sample data files are included

### Performance Optimization

1. **Reduce Dependencies**: Remove unused packages from `requirements.txt`
2. **Optimize Imports**: Use lazy imports where possible
3. **Cache Results**: Implement caching for expensive operations
4. **Limit Dataset Size**: Consider limiting sample data size

## Security Considerations

1. **API Keys**: Never commit API keys to the repository
2. **Data Privacy**: Ensure uploaded data is handled securely
3. **Rate Limiting**: Implement rate limiting for API calls
4. **Input Validation**: Validate all user inputs

## Monitoring and Maintenance

1. **Check Logs**: Regularly monitor Streamlit Cloud logs
2. **Update Dependencies**: Keep packages updated
3. **Monitor Usage**: Track app usage and performance
4. **User Feedback**: Collect and address user feedback

## Sharing Your App

Once deployed, you can share your app using:

- **Direct URL**: `https://your-app-name.streamlit.app`
- **QR Code**: Generate QR code for easy mobile access
- **Social Media**: Share on LinkedIn, Twitter, etc.
- **Documentation**: Include in your README and portfolio

## Cost Considerations

- **Streamlit Cloud**: Free tier available
- **OpenAI API**: Pay-per-use for GPT-4 calls
- **Storage**: Minimal for sample datasets
- **Bandwidth**: Included in free tier

## Support

If you encounter issues:

1. **Check Streamlit Cloud logs**
2. **Review GitHub issues**
3. **Test locally first**
4. **Contact Streamlit support**

---

**🎉 Congratulations! Your ML Pipeline Orchestrator is now live and ready to share with the world!**
