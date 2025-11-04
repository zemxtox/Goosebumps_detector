# CHILLER - Goosebump Detection System

A real-time goosebump detection system that uses computer vision and FFT analysis to detect and visualize goosebumps from video feeds.

## Features

- Real-time goosebump detection using grayscale image processing
- Interactive dashboard with live visualization
- Support for multiple devices/clients simultaneously
- Video file upload and processing
- Frame-by-frame analysis with intensity tracking

## Deployment on Railway

### Prerequisites

1. Create a [Railway](https://railway.app/) account
2. Install the Railway CLI (optional for command-line deployment)

### Deployment Steps

#### Option 1: Deploy via Railway Dashboard

1. Fork this repository to your GitHub account
2. Log in to your Railway account
3. Click "New Project" and select "Deploy from GitHub repo"
4. Select your forked repository
5. Railway will automatically detect the configuration files and deploy the application
6. Once deployed, Railway will provide you with a URL to access your application

#### Option 2: Deploy via Railway CLI

1. Install the Railway CLI: `npm i -g @railway/cli`
2. Login to your Railway account: `railway login`
3. Initialize a new project: `railway init`
4. Deploy the application: `railway up`

### Environment Variables

The following environment variables can be configured in Railway:

- `PORT`: The port on which the server will run (default: 8000)
- `HOST`: The host address (default: 0.0.0.0)

## Local Development

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd goosebumps_detector

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
python chiller.py
```

The application will be available at `http://localhost:8000`.

## Usage

1. Open the dashboard in your browser
2. Allow camera access or upload a video file
3. The system will establish a baseline and then begin detecting goosebumps
4. View real-time detection results and intensity graphs

## License

This project is licensed under the MIT License - see the LICENSE file for details.