# Railway Deployment Guide for CHILLER

## Quick Deploy to Railway

### Option 1: Deploy from GitHub (Recommended)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Railway deployment"
   git push origin main
   ```

2. **Deploy on Railway**:
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will automatically detect and deploy your app

### Option 2: Deploy with Railway CLI

1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and Deploy**:
   ```bash
   railway login
   railway init
   railway up
   ```

## Environment Variables (Optional)

Set these in Railway dashboard under "Variables":

```
DETECTION_THRESHOLD=25.0
VIDEO_DETECTION_THRESHOLD=60.0
BASELINE_FRAMES=5
SAVE_DETECTIONS=true
SECRET_KEY=your-secret-key-here
```

## Features Available on Railway

✅ **Full PWA Support**: Railway provides HTTPS automatically
✅ **Camera Access**: Works with HTTPS
✅ **Service Worker**: Caching and offline support
✅ **App Installation**: Install as native app
✅ **Real-time Detection**: WebSocket support
✅ **File Upload**: Video processing
✅ **Mobile Optimized**: Responsive design

## Post-Deployment Checklist

1. **Test PWA Installation**:
   - Visit your Railway URL
   - Look for install prompt in browser
   - Test "Install App" button

2. **Test Camera Access**:
   - Click "Detect from Camera"
   - Allow camera permissions
   - Verify video feed appears

3. **Test Real-time Detection**:
   - Upload a test video or use camera
   - Verify detection results appear
   - Check live graphs update

## Troubleshooting

### App won't install as PWA
- Ensure you're using the Railway HTTPS URL
- Check browser console for manifest errors
- Try refreshing the page

### Camera not working
- Verify you're using HTTPS (Railway URL)
- Check camera permissions in browser
- Try different browser (Chrome recommended)

### WebSocket connection issues
- Check Railway logs for errors
- Verify CORS settings in deployment
- Try refreshing the page

## Railway-Specific Features

- **Automatic HTTPS**: No SSL certificate setup needed
- **Custom Domain**: Add your own domain in Railway dashboard
- **Auto-scaling**: Handles traffic spikes automatically
- **Logs**: View real-time logs in Railway dashboard
- **Metrics**: Monitor CPU, memory, and network usage

## File Structure for Railway

```
├── chiller.py              # Main Flask application
├── chiller_dashboard.html  # Frontend interface
├── manifest.json          # PWA manifest
├── service-worker.js      # Service worker for PWA
├── requirements.txt       # Python dependencies
├── Procfile              # Railway process configuration
├── railway.json          # Railway deployment config
├── icons/                # PWA icons
└── README.md            # This file
```

## Performance Optimization

Railway deployment includes:
- Optimized image processing (640px max)
- Frame skipping for better performance
- Efficient WebSocket communication
- Compressed JPEG encoding
- Smart caching with Service Worker

## Support

If you encounter issues:
1. Check Railway deployment logs
2. Verify all files are committed to git
3. Ensure requirements.txt is up to date
4. Test locally first with `python chiller.py`

## Local Development vs Railway

| Feature | Local | Railway |
|---------|-------|---------|
| HTTPS | Manual setup | Automatic |
| PWA Install | localhost only | Full support |
| Camera Access | localhost only | Full support |
| Custom Domain | No | Yes |
| SSL Certificate | Self-signed | Valid CA |
| Scaling | Manual | Automatic |