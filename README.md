# CCTV Violence & Harassment Detection System

An AI-powered web application that analyzes CCTV footage to detect violent or suspicious activities automatically.  
The system processes video frames using a trained computer vision model and identifies incidents such as violence or harassment.

## Features

- Upload CCTV video footage
- Detect violence and harassment using AI
- Automatic frame extraction from videos
- Incident timeline with timestamps
- Bounding box predictions from the AI model
- Confidence score for each detection
- Incident clustering to avoid false alarms
- Export results as CSV

## How It Works

1. User uploads a CCTV video.
2. The backend extracts frames at regular intervals.
3. Frames are sent to a trained Roboflow model for prediction.
4. The system detects violent behavior or harassment.
5. Detected incidents are grouped and displayed with timestamps.

## Tech Stack

Frontend:
- HTML
- CSS
- JavaScript

Backend:
- Python
- Flask
- OpenCV

AI Model:
- Roboflow Computer Vision Model

Deployment:
- Render / Flask Server

## Project Structure
