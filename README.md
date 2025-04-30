# 🥂 SipSync

SipSync is a creative AI-powered application that generates custom cocktail recipes based on songs. By analyzing the mood of a song, SipSync prompts a language model to generate a cocktail recipe that matches the vibe of the music.

---

## 🎯 Project Goals

- Predict the mood of a song using machine learning
- Generate a cocktail recipe that reflects the mood of the song
- Build an end-to-end pipeline from song input to recipe output
- Develop both backend model training and a user-friendly front-end app

---

## 📊 Data Used

We use the **MTG-Jamendo dataset**, a large-scale open-source music dataset consisting of:

- 55,000+ Creative Commons-licensed audio tracks  
- 195 tags across genre, instrument, and mood/theme categories  



---

## 📁 Project Structure

The codebase is organized into two main directories:

SipSyncModels/
Contains scripts and notebooks for:

Loading and preprocessing the Jamendo data

Training and fine-tuning a mood classification model using transfer learning

SipSyncGUI/
Contains the front-end code that:

Accepts a song name input

Searches YouTube and downloads the audio

Runs the mood classifier

Uses the predicted mood to generate a cocktail recipe using OpenAI's API

```bash
cd SipSyncModels
# Open and run the provided Jupyter Notebook called SipSync2
```
```bash
cd SipSyncGUI
python app.py

```

---

## 👥 Team Contribution
Ivana Rasch: Data loading and access, front-end development including GUI and audio pipeline

Kelvin Walls: Data cleaning, preprocessing, model development and training


