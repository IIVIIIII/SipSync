# SipSync

🎵🍸 Project Idea and Goals
SipSync is a creative AI-powered project that generates custom cocktail recipes based on songs. The goal is to explore the relationship between music and mood, using machine learning to classify a song’s emotional profile and then prompting a generative model to produce a thematically aligned cocktail recipe.

📊 Data Used
We used the MTG-Jamendo dataset, a large open-source collection of over 55,000 Creative Commons-licensed music tracks. Each track is tagged with genre, instrument, and mood/theme annotations. For this project, we focused on the mood/theme subset (approximately 18,000 tracks) to train a model that predicts the emotional tone of a song.

🗂 Code Structure
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

To run the backend model:

bash
Copy
Edit
cd SipSyncModels  
Run the Jupyter Notebook provided  
To test the front-end (locally):

bash
Copy
Edit
cd SipSyncGUI  
python app.py  
👥 Team Contribution
Ivana Rasch: Data loading and access, full front-end development including GUI and audio pipeline

Kelvin [Last Name]: Data cleaning, preprocessing, model development and training
