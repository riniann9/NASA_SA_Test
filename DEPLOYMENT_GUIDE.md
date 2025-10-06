# Render Deployment Guide

## Overview
This guide will help you deploy your NASA Space Apps Challenge exoplanet predictor to Render.

## Prerequisites
1. A GitHub account with your code repository
2. A Render account (sign up at [render.com](https://render.com))
3. A Gemini API key (optional, for AI analysis features)

## Deployment Steps

### 1. Push to GitHub
Make sure your code is pushed to a GitHub repository:
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Connect to Render
1. Go to [render.com](https://render.com) and sign in
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Select your repository

### 3. Configure the Service
Use these settings:
- **Name**: `exoplanet-predictor` (or your preferred name)
- **Environment**: `Node`
- **Plan**: `Free` (or upgrade if needed)
- **Build Command**: `cd exoplanet-predictor && npm install && npm run build`
- **Start Command**: `cd exoplanet-predictor && npm start`
- **Root Directory**: `exoplanet-predictor`

### 4. Environment Variables
Add these environment variables in Render dashboard:
- `NODE_ENV`: `production`
- `GEMINI_API_KEY`: `your_actual_gemini_api_key_here` (optional)

### 5. Deploy
Click "Create Web Service" and wait for deployment to complete.

## Alternative: Using render.yaml
If you prefer automatic configuration:
1. The `render.yaml` file is already created in your project root
2. Connect your GitHub repository to Render
3. Render will automatically detect and use the configuration

## Getting a Gemini API Key (Optional)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Add it to your Render environment variables

## Features Available After Deployment
- ✅ Exoplanet prediction interface
- ✅ Light curve analysis
- ✅ Planet visualization
- ✅ AI-powered analysis (with Gemini API key)
- ✅ Responsive design
- ✅ Modern UI with space theme

## Troubleshooting
- If build fails, check the build logs in Render dashboard
- Ensure all dependencies are in package.json
- Verify environment variables are set correctly
- Check that the root directory is set to `exoplanet-predictor`

## Custom Domain (Optional)
You can add a custom domain in the Render dashboard under "Settings" → "Custom Domains".

## Monitoring
- View logs in the Render dashboard
- Monitor performance and usage
- Set up alerts for downtime

Your app will be available at: `https://your-app-name.onrender.com`
