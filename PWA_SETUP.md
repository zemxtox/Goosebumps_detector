# PWA Setup Guide for CHILLER Goosebump Detector

## Quick Start

### Option 1: HTTPS (Recommended for full PWA features)
```bash
# Windows
run_https.bat

# Manual
python generate_cert.py
python chiller_https.py
```
Then visit: https://localhost:8000

### Option 2: HTTP (Limited features)
```bash
python chiller.py
```
Then visit: http://localhost:8000

## PWA Installation

### Desktop (Chrome/Edge)
1. Visit the app in your browser
2. Look for the "Install" button in the address bar
3. Or click the "Install App" button in the header
4. Follow the installation prompts

### Mobile (Android)
1. Open the app in Chrome
2. Tap the menu (⋮) 
3. Tap "Add to Home screen" or "Install app"
4. Follow the prompts

### Mobile (iOS)
1. Open the app in Safari
2. Tap the Share button (□↑)
3. Scroll down and tap "Add to Home Screen"
4. Tap "Add"

## Features Available

| Feature | HTTP | HTTPS |
|---------|------|-------|
| Basic UI | ✅ | ✅ |
| Video Upload | ✅ | ✅ |
| Camera Access | ❌ | ✅ |
| PWA Installation | ❌ | ✅ |
| Offline Support | ❌ | ✅ |
| Push Notifications | ❌ | ✅ |

## Troubleshooting

### "Service Worker not supported"
- Use HTTPS or localhost
- Make sure you're not in incognito/private mode

### "Camera access requires HTTPS"
- Use the HTTPS version: `python chiller_https.py`
- Or use localhost: `http://localhost:8000`

### "Install button not showing"
- Make sure you're using HTTPS
- Try refreshing the page
- Check browser console for errors

### Certificate warnings (HTTPS)
- This is normal for self-signed certificates
- Click "Advanced" → "Proceed to localhost (unsafe)"
- The warning only appears once per browser

## Browser Support

- ✅ Chrome 67+
- ✅ Firefox 60+
- ✅ Safari 11.1+
- ✅ Edge 79+

## Development Notes

- Service Worker caches static assets for offline use
- Camera requires secure context (HTTPS or localhost)
- PWA installation requires HTTPS and valid manifest
- Self-signed certificates are fine for development