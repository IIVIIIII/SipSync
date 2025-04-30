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

For this project, we focus on the **mood/theme subset** (~18,000 tracks), which is suitable for mood classification tasks.

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

