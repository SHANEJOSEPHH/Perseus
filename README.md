<img width="1280" height="640" alt="git (1)" src="https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd" />



# [OVERTHINKING.AI] 🎯


## Basic Details
### Team Name: [PERSEUS]


### Team Members
- Team Lead: [Shane Joseph] - [AISAT]

### Project Description
[OVERTHINKING.AI is a local AI-powered simulator that analyzes everyday situations people tend to overthink. It generates multiple possible explanations, calculates an Overthinking Score, and gives a final evidence-based verdict.]

### The Problem (that doesn't exist)
[Someone says "okay" instead of "okayyy", replies three hours later, sits somewhere else, or views your story without replying.

Suddenly, your brain starts running a full investigation.

Was something wrong? Were they angry? Did I do something? Are they avoiding me?

We decided this completely unnecessary problem deserved completely unnecessary technology.]

### The Solution (that nobody asked for)
[We built OVERTHINKING.AI.

Users enter a situation they're overthinking, and the system uses a local AI model to:

*Generate five possible explanations.
*Consider the actual facts and timeline.
*Avoid treating assumptions as facts.
*Detect contradictions in AI-generated explanations.
*Calculate an Overthinking Score.
*Provide a final evidence-based verdict.

Basically, we gave your overthinking a data-processing department.]

## Technical Details
### Technologies/Components Used
For Software:
- [Python — AI/backend logic and Flask server
HTML — Webpage structure
CSS — UI design and styling
JavaScript — Frontend interaction and API communication]
- [Flask — Python web framework for the backend and API
Ollama — Local AI runtime for running the Qwen 2.5 Coder 3B model]
- [Flask — Web server and API handling
Requests — Communicates with the local Ollama AI API
re (Regular Expressions) — Fact extraction and contradiction checking
JSON — Handles AI responses and API data]
- [Visual Studio Code — Code development and editing
Ollama — Running the local AI model
Git & GitHub — Version control and project hosting
Command Prompt / Terminal — Installing dependencies and running the application
Web Browser — Testing and interacting with the simulator]

### Implementation
For Software:
# Installation
[git clone [https://github.com/SHANEJOSEPHH/Perseus]
cd overthinking-simulator]
[pip install flask requests]
[ollama pull qwen2.5-coder:3b]
Start Ollama before starting the Flask application.

# Run
[python app.py]
[http://127.0.0.1:5000]Open this address in a web browser and enter something to overthink.

### Project Documentation
For Software:

# Screenshots (Add at least 3)
![main-interface.png](Main Interface)
*The main OVERTHINKING.AI interface where users enter a situation they are overthinking*

![ai-analysis.png](AI Analysis)
*The system generates five possible explanations using the local AI model and assigns an AI weight to each interpretation.*

![final-verdict.png](Final Verdict)
*The final screen displays the Overthinking Score and an evidence-based verdict based on the available information.*

# Diagrams
![Workflow](User Input➡️Flask Backend➡️Ollama Local AI➡️Qwen 2.5 Coder 3B➡️5 Possible Explanations➡️Evidence & Contradiction Validation➡️Overthinking Score➡️Final Verdict➡️Results displayed in the Web Interface)

---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--26-26?link=https%3A%2F%2Ftinkerhub.org%2Fevents%2F1M8ORET9A1%2Fuseless-projects-3.0)



