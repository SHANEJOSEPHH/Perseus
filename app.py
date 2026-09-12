from flask import Flask, render_template, request, jsonify
import requests
import re
import random

app = Flask(__name__)

# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:3b"


# ============================================================
# AI SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the reasoning engine for OVERTHINKING.AI.

Your job is to analyze a user's situation and generate five genuinely
different possible interpretations.

IMPORTANT:

1. READ THE ENTIRE SITUATION CAREFULLY.
2. Preserve exact qualifiers and relationships.
3. Words such as:
   extra, already, only, didn't, never, always, before, after,
   because, but, however, another, both, neither, etc.
   are important facts.
4. Never contradict or silently change a fact.
5. Do not invent people, objects, owners, locations, noises,
   conversations, previous events, or background circumstances.
6. Reason from the information actually given.
7. Reasonable inferences are allowed.
8. Do not simply repeat the user's sentence.
9. Do not make every explanation mean the same thing.
10. Each explanation MUST explore a meaningfully different angle.
11. Do not repeat the same cause using different wording.
12. When possible, vary the reasoning angle:
    - possible cause
    - emotional/instinctive reaction
    - purpose of the action
    - alternative interpretation
    - what the evidence does NOT establish
13. If the situation does not provide enough evidence for a specific cause,
    explicitly recognize that uncertainty instead of inventing a cause.
OUTPUT EXACTLY THIS FORMAT:

SCORE: number from 0 to 100

1. explanation
2. explanation
3. explanation
4. explanation
5. explanation

VERDICT: one concise conclusion
"""


# ============================================================
# FALLBACKS
# ============================================================

# These are only used if Ollama completely fails.
# They are intentionally connected to the user's situation.

def fallback_results(situation):
    return [
        {
            "text": "The situation may have a simpler explanation than the first interpretation suggests.",
            "probability": 30
        },
        {
            "text": "The available details leave room for more than one reasonable interpretation.",
            "probability": 25
        },
        {
            "text": "One of the actions described may have a reason that is not obvious from the situation alone.",
            "probability": 20
        },
        {
            "text": "Your interpretation may depend partly on what you expected the other side to do.",
            "probability": 15
        },
        {
            "text": "There may not be enough evidence in the situation to confidently reach the strongest conclusion.",
            "probability": 10
        }
    ]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\r", "")
    text = text.replace("```text", "")
    text = text.replace("```", "")

    return text.strip()


def clean_interpretation(text):
    text = text.strip()

    # Remove accidental numbering
    text = re.sub(r"^\s*\d+\s*[\.\):-]\s*", "", text)

    # Remove accidental labels
    text = re.sub(
        r"^\s*interpretation\s*\d*\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^\s*possible explanation\s*\d*\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.strip(" -•")

    return text


# ============================================================
# SCORE EXTRACTION
# ============================================================

def extract_score(text):
    patterns = [
        r"SCORE\s*:\s*(\d{1,3})",
        r"OVERTHINKING\s*SCORE\s*:\s*(\d{1,3})",
        r"(\d{1,3})\s*/\s*100"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            score = int(match.group(1))
            return max(0, min(100, score))

    return random.randint(35, 75)


# ============================================================
# EXPLANATION EXTRACTION
# ============================================================

def extract_interpretations(text):
    explanations = []

    # --------------------------------------------------------
    # First try numbered lines
    # --------------------------------------------------------

    lines = text.splitlines()

    for line in lines:
        line = line.strip()

        match = re.match(
            r"^\s*(\d+)\s*[\.\):-]\s*(.+)$",
            line
        )

        if match:
            number = int(match.group(1))
            explanation = clean_interpretation(match.group(2))

            if 1 <= number <= 5 and explanation:
                explanations.append(explanation)

    # --------------------------------------------------------
    # Try "Interpretation 1:"
    # --------------------------------------------------------

    if len(explanations) < 5:

        pattern = re.compile(
            r"Interpretation\s*(\d+)\s*[:\-]\s*(.+?)(?="
            r"\n\s*Interpretation\s*\d+"
            r"|\n\s*\d+\s*[\.\):-]"
            r"|\n\s*VERDICT"
            r"|$)",
            re.IGNORECASE | re.DOTALL
        )

        matches = pattern.findall(text)

        for number, explanation in matches:
            number = int(number)

            if 1 <= number <= 5:
                explanation = clean_interpretation(explanation)

                if explanation:
                    explanations.append(explanation)

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique = []

    for explanation in explanations:

        normalized = re.sub(
            r"\s+",
            " ",
            explanation.lower()
        ).strip()

        if normalized not in [
            re.sub(r"\s+", " ", x.lower()).strip()
            for x in unique
        ]:
            unique.append(explanation)

    return unique[:5]


# ============================================================
# VERDICT EXTRACTION
# ============================================================

def extract_verdict(text):
    match = re.search(
        r"VERDICT\s*:\s*(.+)",
        text,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        verdict = match.group(1).strip()

        # Remove accidental trailing formatting
        verdict = verdict.split("\n\n")[0].strip()

        return verdict

    return "The evidence does not fully support the strongest conclusion."


# ============================================================
# UNSUPPORTED CONCRETE DETAILS
# ============================================================

UNSUPPORTED_CONTEXT_PHRASES = [

    # People
    "someone else nearby",
    "someone nearby",
    "another person nearby",
    "another person",
    "a stranger",
    "someone else",

    # Owners
    "its owner",
    "their owner",
    "his owner",
    "her owner",
    "the owner",

    # Sounds
    "a sudden noise",
    "a loud noise",
    "background noise",
    "a sound nearby",

    # Unmentioned history
    "recent events",
    "previous events",
    "past events",
    "previous experience",
    "past experience",

    # Locations/environment
    "new environment",
    "unfamiliar surroundings",
    "different environment",

    # Other invented circumstances
    "someone scared",
    "someone frightened",
    "someone called",
    "someone shouted",
    "the owner called",
    "the owner shouted"
]


# ============================================================
# VALIDATION
# ============================================================

def validate_interpretation(situation, explanation):
    """
    Validate whether an explanation contains clearly invented
    concrete circumstances.

    IMPORTANT:
    We deliberately DO NOT require literal word overlap.

    Example:
        "The cat may have perceived you as a threat."

    is valid even if "threat" was never explicitly written.

    But:
        "The cat may have been scared by its owner."

    is invalid if no owner was mentioned.
    """

    explanation_lower = explanation.lower()

    # --------------------------------------------------------
    # 1. Reject generic filler
    # --------------------------------------------------------

    generic_phrases = [
        "there may be a completely ordinary explanation",
        "the situation could mean less than it seems",
        "another detail may change",
        "you may be overthinking a moment",
        "your imagination may be influencing",
        "the situation may simply be ordinary"
    ]

    for phrase in generic_phrases:

        if phrase in explanation_lower:
            return False, "Generic explanation."

    # --------------------------------------------------------
    # 2. Reject clearly invented concrete context
    # --------------------------------------------------------

    for phrase in UNSUPPORTED_CONTEXT_PHRASES:

        if phrase in explanation_lower:

            # Only reject if the situation itself does not
            # contain the relevant concept.

            situation_lower = situation.lower()

            concept_words = phrase.split()

            relevant_found = False

            for word in concept_words:

                word = word.strip(".,!?")

                if len(word) >= 4 and word in situation_lower:
                    relevant_found = True
                    break

            if not relevant_found:
                return False, (
                    f"Unsupported detail: '{phrase}'"
                )

    # --------------------------------------------------------
    # 3. Reject very short / useless explanations
    # --------------------------------------------------------

    if len(explanation.split()) < 6:
        return False, "Explanation is too short."

    # --------------------------------------------------------
    # 4. Reject explanations that are basically the input
    # --------------------------------------------------------

    normalized_situation = re.sub(
        r"[^a-z0-9\s]",
        "",
        situation.lower()
    )

    normalized_explanation = re.sub(
        r"[^a-z0-9\s]",
        "",
        explanation.lower()
    )

    situation_words = set(normalized_situation.split())
    explanation_words = set(normalized_explanation.split())

    if len(explanation_words) > 0:

        overlap = len(
            situation_words.intersection(explanation_words)
        )

        ratio = overlap / len(explanation_words)

        if ratio > 0.85 and len(explanation_words) > 8:
            return False, "Explanation mostly repeats the input."

    return True, ""


def validate_interpretations(situation, explanations):
    problems = []

    if len(explanations) != 5:
        problems.append(
            f"Expected 5 explanations, got {len(explanations)}."
        )

    for i, explanation in enumerate(explanations, start=1):

        valid, reason = validate_interpretation(
            situation,
            explanation
        )

        if not valid:
            problems.append(
                f"Explanation {i}: {reason}"
            )

    return problems


# ============================================================
# AI RESULT CREATION
# ============================================================

def create_results(explanations):
    weights = [30, 25, 20, 15, 10]

    results = []

    for i, explanation in enumerate(explanations[:5]):

        weight = weights[i]

        results.append({
            "text": explanation,
            "reason": explanation,
            "title": explanation,

            "probability": weight,
            "confidence": weight,
            "weight": weight
        })

    return results


# ============================================================
# AI ANALYSIS
# ============================================================

def analyze_with_ai(situation, correction=None):

    if correction:

        prompt = f"""
Analyze this exact situation:

{situation}

Some previous explanations were rejected because they contained
unsupported details or generic reasoning.

Generate five NEW specific interpretations.

Use only information supported by the situation.
Reasonable inference is allowed.
Do not invent people, owners, noises, locations, previous events,
or background circumstances.

Do not repeat the facts.

Output exactly:

SCORE: number
1. explanation
2. explanation
3. explanation
4. explanation
5. explanation
VERDICT: conclusion
"""

    else:

        prompt = f"""
{SYSTEM_PROMPT}

USER SITUATION:

{situation}

Now analyze it.
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,

        "options": {
            "temperature": 0.65,
            "num_predict": 180
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=45
        )

        response.raise_for_status()

        data = response.json()

        raw = data.get("response", "")

        raw = clean_text(raw)

        return raw

    except Exception as e:

        print("\nOLLAMA ERROR:")
        print(e)

        return ""


# ============================================================
# COMPLETE ANALYSIS PIPELINE
# ============================================================

def analyze_situation(situation):

    situation = situation.strip()

    if not situation:
        return {
            "score": 0,
            "results": [],
            "verdict": "Please enter a situation first.",
            "level": "UNKNOWN"
        }

    # ========================================================
    # FIRST AI PASS
    # ========================================================

    print("\n" + "=" * 60)
    print("AI RESPONSE - FIRST PASS")
    print("=" * 60)

    raw = analyze_with_ai(situation)

    print(raw)

    score = extract_score(raw)

    explanations = extract_interpretations(raw)

    verdict = extract_verdict(raw)

    # ========================================================
    # VALIDATE FIRST PASS
    # ========================================================

    problems = validate_interpretations(
        situation,
        explanations
    )

    if problems:

        print("\n" + "=" * 60)
        print("PYTHON CONSISTENCY CHECK FAILED")
        print("=" * 60)

        for problem in problems:
            print("-", problem)

        # ====================================================
        # REGENERATE
        # ====================================================

        print("\n" + "=" * 60)
        print("AI REGENERATION")
        print("=" * 60)

        regenerated = analyze_with_ai(
            situation,
            correction=problems
        )

        print(regenerated)

        new_score = extract_score(regenerated)

        new_explanations = extract_interpretations(
            regenerated
        )

        new_verdict = extract_verdict(
            regenerated
        )

        new_problems = validate_interpretations(
            situation,
            new_explanations
        )

        # ----------------------------------------------------
        # Use regenerated response if valid
        # ----------------------------------------------------

        if not new_problems:

            print("\nREGENERATION PASSED.")

            score = new_score
            explanations = new_explanations
            verdict = new_verdict

        else:

            print("\nREGENERATION STILL FAILED.")

            for problem in new_problems:
                print("-", problem)

    else:

        print("\n" + "=" * 60)
        print("PYTHON CONSISTENCY CHECK PASSED")
        print("=" * 60)

    # ========================================================
    # FINAL SAFETY NET
    # ========================================================

    if len(explanations) < 5:

        print("\nAI DID NOT RETURN FIVE VALID EXPLANATIONS.")

        # Try one final compact request.
        final_raw = analyze_with_ai(
            situation,
            correction=[
                "Return exactly five numbered explanations.",
                "Keep them specific to the situation.",
                "Do not invent concrete circumstances."
            ]
        )

        print("\n" + "=" * 60)
        print("FINAL AI ATTEMPT")
        print("=" * 60)

        print(final_raw)

        final_score = extract_score(final_raw)

        final_explanations = extract_interpretations(
            final_raw
        )

        final_verdict = extract_verdict(
            final_raw
        )

        if len(final_explanations) == 5:

            score = final_score
            explanations = final_explanations
            verdict = final_verdict

    # ========================================================
    # LAST RESORT
    # ========================================================

    if len(explanations) < 5:

        print("\nUSING MINIMAL FALLBACK.")

        fallback = fallback_results(situation)

        explanations = [
            item["text"]
            for item in fallback
        ]

        verdict = (
            "The available evidence supports several possibilities, "
            "but not a definite conclusion."
        )

    # ========================================================
    # SCORE
    # ========================================================

    score = max(
        0,
        min(
            100,
            int(score)
        )
    )

    # ========================================================
    # LEVEL
    # ========================================================

    if score < 25:
        level = "LOW"

    elif score < 50:
        level = "MILD"

    elif score < 70:
        level = "MODERATE"

    elif score < 85:
        level = "HIGH"

    else:
        level = "EXTREME"

    # ========================================================
    # RESULTS
    # ========================================================

    results = create_results(
        explanations
    )

    print("\n" + "=" * 60)
    print("FINAL ANALYSIS")
    print("=" * 60)

    print("SCORE:", score)
    print("LEVEL:", level)

    for i, explanation in enumerate(
        explanations,
        start=1
    ):
        print(f"{i}. {explanation}")

    print("VERDICT:", verdict)

    print("=" * 60)

    return {
        "score": score,
        "results": results,
        "verdict": verdict,
        "level": level
    }


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No data received."
            }), 400

        situation = data.get(
            "situation",
            ""
        ).strip()

        if not situation:

            return jsonify({
                "error": "Please enter a situation."
            }), 400

        result = analyze_situation(
            situation
        )

        return jsonify(result)

    except Exception as e:

        print("\nSERVER ERROR:")
        print(e)

        return jsonify({
            "error": "Something went wrong while analyzing the situation."
        }), 500


# ============================================================
# START SERVER
# ============================================================

@app.route("/esp32", methods=["GET"])
def esp32():

    print("ESP32-S3 connected!")

    return jsonify({
        "status": "ESP32 ONLINE",
        "message": "OVERTHINKING.AI hardware connected"
    })

if __name__ == "__main__":

    print("=" * 60)
    print("OVERTHINKING.AI")
    print("Local AI Engine:", MODEL)
    print("Server: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )