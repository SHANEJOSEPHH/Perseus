from flask import Flask, render_template, request, jsonify
import requests
import re
import json

app = Flask(__name__)

# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5-coder:3b"

AI_TIMEOUT = 55

WEIGHTS = [30, 25, 20, 15, 10]


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = str(text)
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_json(text):
    """
    Extract JSON from Ollama output even if the model adds
    markdown fences or extra text.
    """

    text = text.strip()

    # Remove markdown fences
    text = re.sub(r"```json\s*", "", text, flags=re.I)
    text = re.sub(r"```\s*", "", text)

    # Direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Find JSON array
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


# ============================================================
# FACT EXTRACTION
# ============================================================

def get_facts(situation):
    """
    Extract ONLY facts that are explicitly present.

    This is intentionally conservative. It helps prevent the
    small local model from inventing context.
    """

    s = situation.lower()

    facts = {
        "online": False,
        "posted": False,
        "normal_after": False,
        "no_change": False,
        "didnt_reply": False,
        "later_replied": False,
        "story_not_about_user": False,
        "no_mention": False,

        # New conservative signals
        "short_reply": False,
        "okay_reply": False,
        "delayed_reply": False,
        "multiple_messages": False,
        "explicit_teacher": False,
        "explicit_friend": False,
        "explicit_partner": False,
    }

    # Online
    if re.search(
        r"\bonline\b|\bactive\b|\bactive now\b|\bonline now\b",
        s
    ):
        facts["online"] = True

    # Posted
    if re.search(
        r"\bposted\b|\bpost\b|\buploaded\b|\bput up a post\b",
        s
    ):
        facts["posted"] = True

    # Normal after
    if re.search(
        r"\bacted normal\b|\bwas normal\b|\bnormal afterwards\b|"
        r"\bnormal after\b|\bbehaved normally\b",
        s
    ):
        facts["normal_after"] = True

    # No change
    if re.search(
        r"\bnothing changed\b|\bno change\b|\beverything stayed normal\b",
        s
    ):
        facts["no_change"] = True

    # Didn't reply
    if re.search(
        r"\bdidn'?t reply\b|\bdid not reply\b|\bnever replied\b|"
        r"\bhasn't replied\b|\bhas not replied\b",
        s
    ):
        facts["didnt_reply"] = True

    # Later replied
    if re.search(
        r"\blater replied\b|\breplied later\b|\beventually replied\b|"
        r"\breplied afterwards\b|\breplied after\b",
        s
    ):
        facts["later_replied"] = True

    # Story explicitly not about user
    if re.search(
        r"\bstory wasn'?t about me\b|\bstory was not about me\b|"
        r"\bstory wasn't about me\b|\bstory had nothing to do with me\b|"
        r"\bstory was not related to me\b",
        s
    ):
        facts["story_not_about_user"] = True

    # No mention
    if re.search(
        r"\bdidn'?t mention me\b|\bdid not mention me\b|"
        r"\bwithout mentioning me\b|\bnever mentioned me\b",
        s
    ):
        facts["no_mention"] = True

    # Short reply
    if re.search(
        r"\bonly\s+['\"]?.{1,30}['\"]?\s*(?:to|as)\b|"
        r"\bjust\s+replied\b|\bonly replied\b|\bshort reply\b|"
        r"\bshort response\b",
        s
    ):
        facts["short_reply"] = True

    # Okey / Okay
    if re.search(
        r"\bokay\b|\bok\b|\bokey\b|\bokeyy\b|\bokayy\b",
        s
    ):
        facts["okay_reply"] = True

    # Delayed reply — ONLY if explicitly stated
    if re.search(
        r"\bdelayed reply\b|\breplied late\b|\blate reply\b|"
        r"\btook .* to reply\b|\breplied after .* minutes\b|"
        r"\breplied after .* hours\b",
        s
    ):
        facts["delayed_reply"] = True

    # Multiple messages — ONLY if explicitly stated
    if re.search(
        r"\bmultiple messages\b|\bseveral messages\b|\bmany messages\b|"
        r"\bsent .* messages\b|\btexted multiple times\b",
        s
    ):
        facts["multiple_messages"] = True

    # Explicit relationships
    if re.search(r"\bmy teacher\b|\bmy teachers\b", s):
        facts["explicit_teacher"] = True

    if re.search(r"\bmy friend\b|\bmy friends\b", s):
        facts["explicit_friend"] = True

    if re.search(
        r"\bmy boyfriend\b|\bmy girlfriend\b|\bmy partner\b|"
        r"\bmy husband\b|\bmy wife\b",
        s
    ):
        facts["explicit_partner"] = True

    return facts


# ============================================================
# INVENTED CONTEXT DETECTION
# ============================================================

INVENTED_CONTEXT_PATTERNS = [
    # Work / school / activities
    r"\bproject\b",
    r"\bmeeting\b",
    r"\bclass\b",
    r"\bhomework\b",
    r"\bassignment\b",
    r"\bwork\b",
    r"\boffice\b",
    r"\bschool\b",
    r"\bcollege\b",
    r"\bevent\b",

    # Health
    r"\bnot feeling well\b",
    r"\bfeeling sick\b",
    r"\bsick\b",
    r"\bheadache\b",
    r"\btired\b",
    r"\bexhausted\b",
    r"\bunwell\b",

    # Technical excuses
    r"\btechnical difficult",
    r"\binternet\b",
    r"\bnetwork\b",
    r"\bwifi\b",
    r"\bphone problem\b",
    r"\bphone issue\b",
    r"\bdevice problem\b",
    r"\bbattery\b",

    # Emergencies
    r"\bemergency\b",
    r"\baccident\b",
    r"\bfamily problem\b",
    r"\bfamily emergency\b",

    # Travel
    r"\btravel\b",
    r"\btravelling\b",
    r"\btraveling\b",
    r"\bvaction\b",
    r"\bvacation\b",
    r"\btrip\b",

    # Arguments
    r"\bargument\b",
    r"\bfight\b",
    r"\bdisagreement\b",
    r"\bconflict\b",

    # Emotional assumptions
    r"\bjealous\b",
    r"\bangry\b",
    r"\bupset\b",
    r"\bsad\b",
    r"\bannoyed\b",
    r"\bfrustrated\b",
    r"\bstressed\b",
    r"\boverwhelmed\b",

    # Other invented situations
    r"\bbusy schedule\b",
    r"\bbusy day\b",
    r"\bother task\b",
    r"\banother task\b",
    r"\bsomething came up\b",
    r"\bwas doing something\b",
    r"\bwas occupied\b",
]


def contains_invented_context(text):
    text = text.lower()

    for pattern in INVENTED_CONTEXT_PATTERNS:
        if re.search(pattern, text):
            return True

    return False


# ============================================================
# CONTRADICTION CHECKING
# ============================================================

def contradicts_facts(explanation, facts):
    """
    Reject explanations that directly contradict facts explicitly
    given by the user.
    """

    e = explanation.lower()

    # User explicitly says online
    if facts["online"]:
        bad_patterns = [
            r"\boffline\b",
            r"\bwithout internet\b",
            r"\bno internet\b",
            r"\bno connection\b",
            r"\bnetwork problem\b",
            r"\bphone.*not working\b",
        ]

        if any(re.search(p, e) for p in bad_patterns):
            return True

    # User explicitly says posted
    if facts["posted"]:
        bad_patterns = [
            r"\bcouldn't access social media\b",
            r"\bcould not access social media\b",
            r"\bwasn't using social media\b",
            r"\bwas not using social media\b",
        ]

        if any(re.search(p, e) for p in bad_patterns):
            return True

    # User explicitly says normal afterward
    if facts["normal_after"] or facts["no_change"]:
        bad_patterns = [
            r"\bwas angry\b",
            r"\bwas upset\b",
            r"\bwas furious\b",
            r"\bwas mad\b",
            r"\bwas annoyed\b",
        ]

        if any(re.search(p, e) for p in bad_patterns):
            return True

    # Story wasn't about user
    if facts["story_not_about_user"]:
        bad_patterns = [
            r"\babout you\b",
            r"\babout the user\b",
            r"\btargeting you\b",
            r"\btrying to send you a message\b",
        ]

        if any(re.search(p, e) for p in bad_patterns):
            return True

    # Delayed response explicitly stated
    if facts["delayed_reply"] or facts["later_replied"]:
        bad_patterns = [
            r"\breplied immediately\b",
            r"\bimmediate reply\b",
            r"\breplied instantly\b",
        ]

        if any(re.search(p, e) for p in bad_patterns):
            return True

    return False


# ============================================================
# GENERIC / LOW VALUE DETECTION
# ============================================================

GENERIC_PATTERNS = [
    r"\bsomething might have happened\b",
    r"\bsomething could have happened\b",
    r"\bthere could be many reasons\b",
    r"\banything is possible\b",
    r"\bit is possible that anything\b",
    r"\bvarious reasons\b",
    r"\bmany possibilities\b",
    r"\bfor some reason\b",
]


def is_generic(text):
    t = text.lower().strip()

    if len(t) < 25:
        return True

    for pattern in GENERIC_PATTERNS:
        if re.search(pattern, t):
            return True

    return False


# ============================================================
# RELATIONSHIP VALIDATION
# ============================================================

def relationship_is_invented(text, facts):
    """
    Prevent the AI from suddenly turning 'she' into teacher/friend/etc.
    """

    t = text.lower()

    relationships = [
        ("teacher", facts["explicit_teacher"]),
        ("friend", facts["explicit_friend"]),
        ("boyfriend", facts["explicit_partner"]),
        ("girlfriend", facts["explicit_partner"]),
        ("partner", facts["explicit_partner"]),
        ("husband", facts["explicit_partner"]),
        ("wife", facts["explicit_partner"]),
    ]

    for word, allowed in relationships:
        if re.search(r"\b" + re.escape(word) + r"\b", t):
            if not allowed:
                return True

    return False


# ============================================================
# EXPLANATION VALIDATION
# ============================================================

def validate_explanation(explanation, situation, facts):
    explanation = clean_text(explanation)

    if not explanation:
        return False

    if len(explanation) < 30:
        return False

    if len(explanation) > 350:
        return False

    # Must not refer to "the user"
    if re.search(r"\bthe user\b", explanation.lower()):
        return False

    # Must not speak as AI/personal experience
    if re.search(
        r"\bI experienced\b|\bI think she\b|\bI know why\b",
        explanation,
        re.I
    ):
        return False

    # Invented context
    if contains_invented_context(explanation):
        return False

    # Contradictions
    if contradicts_facts(explanation, facts):
        return False

    # Invented relationship
    if relationship_is_invented(explanation, facts):
        return False

    # Too generic
    if is_generic(explanation):
        return False

    # Dangerous certainty
    certainty_patterns = [
        r"\bshe definitely\b",
        r"\bshe clearly\b",
        r"\bshe was definitely\b",
        r"\bshe is definitely\b",
        r"\bthis proves\b",
        r"\bobviously\b",
        r"\bwithout doubt\b",
        r"\bfor sure\b",
    ]

    for pattern in certainty_patterns:
        if re.search(pattern, explanation.lower()):
            return False

    return True


# ============================================================
# VERDICT VALIDATION
# ============================================================

def safe_verdict(situation):
    """
    Conservative fallback verdict.
    """

    facts = get_facts(situation)

    if facts["okay_reply"]:
        return (
            "There isn't enough evidence to know exactly what she meant "
            "by that reply. A short response such as “Okeyy” can simply "
            "be an acknowledgment, and the message alone does not prove "
            "a specific emotion or intention."
        )

    return (
        "The available evidence does not prove one specific reason for "
        "their behavior. The situation alone is not enough to know their "
        "exact intention."
    )


def validate_verdict(verdict, situation, facts):
    if not verdict:
        return False

    verdict = clean_text(verdict)

    if len(verdict) < 40:
        return False

    if len(verdict) > 500:
        return False

    if contains_invented_context(verdict):
        return False

    if contradicts_facts(verdict, facts):
        return False

    if relationship_is_invented(verdict, facts):
        return False

    if re.search(r"\bthe user\b", verdict.lower()):
        return False

    certainty_patterns = [
        r"\bshe definitely\b",
        r"\bshe clearly\b",
        r"\bhe definitely\b",
        r"\bthey definitely\b",
        r"\bthis proves\b",
        r"\bobviously\b",
        r"\bwithout doubt\b",
        r"\bfor sure\b",
    ]

    for pattern in certainty_patterns:
        if re.search(pattern, verdict.lower()):
            return False

    return True


# ============================================================
# LOCAL AI
# ============================================================

def ask_local_ai(situation):
    facts = get_facts(situation)

    explicit_relationship = []

    if facts["explicit_teacher"]:
        explicit_relationship.append("teacher")

    if facts["explicit_friend"]:
        explicit_relationship.append("friend")

    if facts["explicit_partner"]:
        explicit_relationship.append("partner")

    relationship_text = (
        ", ".join(explicit_relationship)
        if explicit_relationship
        else "NO relationship was explicitly provided"
    )

    prompt = f"""
You are the reasoning engine for OVERTHINKING.AI.

Your job is NOT to invent a story.

USER'S EXACT SITUATION:
{situation}

EXPLICITLY AVAILABLE RELATIONSHIP:
{relationship_text}

STRICT EVIDENCE RULES:

- Use ONLY information explicitly stated in the user's situation.
- NEVER invent missing context.
- NEVER assume the person's relationship to the other person.
- NEVER assume they are a teacher, friend, partner, family member,
  coworker, classmate, etc. unless explicitly stated.
- NEVER invent activities, schedules, health problems, technical
  problems, arguments, emergencies, delays, or background events.
- NEVER assume an emotion such as anger, sadness, jealousy, guilt,
  annoyance, or disinterest unless the situation provides evidence.
- NEVER assume why someone replied, behaved, posted, viewed, or acted.
- Treat every explanation as a POSSIBILITY, never a fact.
- If the situation is ambiguous, say that there is not enough evidence.
- Keep explanations directly connected to the exact words/actions given.
- Do not add details just to make an explanation sound realistic.
- Do not make the explanation longer by inventing circumstances.

PERSPECTIVE RULES:

- Speak directly to the person who submitted the situation.
- Use "you" and "your" when referring to their experience.
- Use "your teacher", "your friend", etc. ONLY when that relationship
  was explicitly stated.
- NEVER say "the user".
- NEVER rewrite the situation using "my" from the person's perspective.
- Do not speak as if you personally experienced the situation.

IMPORTANT:

If the user says someone only replied with a short message, do not
assume the person was busy, sick, working, studying, travelling,
having technical problems, angry, upset, distracted, or dealing with
another situation.

For example:

BAD:
"She may not have been feeling well or was distracted by another task."

Why BAD:
The user never said she was unwell or distracted.

BAD:
"Your friend might have been occupied with a project."

Why BAD:
The user never said she was a friend or had a project.

GOOD:
"She may simply have been acknowledging your message without having
anything else to add."

GOOD:
"She might not have known what else to say, so she kept the reply short."

OUTPUT:

Return ONLY valid JSON.

Use exactly this structure:

{{
  "results": [
    {{
      "title": "Possible explanation",
      "reason": "short explanation"
    }},
    {{
      "title": "Possible explanation",
      "reason": "short explanation"
    }},
    {{
      "title": "Possible explanation",
      "reason": "short explanation"
    }},
    {{
      "title": "Possible explanation",
      "reason": "short explanation"
    }},
    {{
      "title": "Possible explanation",
      "reason": "short explanation"
    }}
  ],
  "verdict": "short evidence-based final verdict"
}}

Generate exactly 5 different explanations.

The explanations should be realistic but must NEVER introduce facts
that are not present in the original situation.
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.35,
            "top_p": 0.8,
            "num_predict": 220
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=AI_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()
        raw = data.get("response", "")

        parsed = extract_json(raw)

        if not parsed:
            return None

        return parsed

    except Exception as e:
        print("OLLAMA ERROR:", e)
        return None


# ============================================================
# REPAIR AI
# ============================================================

def repair_explanations(situation, bad_results):
    """
    Second pass for explanations rejected by the validation layer.
    """

    prompt = f"""
You are repairing explanations for OVERTHINKING.AI.

ORIGINAL SITUATION:
{situation}

The following explanations were rejected because they may have
invented information:

{json.dumps(bad_results, ensure_ascii=False)}

STRICT REPAIR RULES:

- The original situation is the ONLY source of facts.
- Do NOT add facts that are not explicitly present.
- Do NOT assume relationships.
- Do NOT assume emotions.
- Do NOT assume reasons for behavior.
- Do NOT invent delays, activities, problems, schedules, conversations,
  health issues, technical issues, or background events.
- Use "you" and "your" for the person who submitted the situation.
- Only mention "teacher", "friend", "partner", etc. if that exact
  relationship appears in the original situation.
- Every explanation must be a possibility, never a confirmed fact.
- If there is insufficient evidence, explicitly say so.
- Keep explanations short and directly tied to the situation.

Return ONLY valid JSON:

{{
  "results": [
    {{
      "title": "Possible explanation",
      "reason": "..."
    }},
    {{
      "title": "Possible explanation",
      "reason": "..."
    }},
    {{
      "title": "Possible explanation",
      "reason": "..."
    }},
    {{
      "title": "Possible explanation",
      "reason": "..."
    }},
    {{
      "title": "Possible explanation",
      "reason": "..."
    }}
  ],
  "verdict": "..."
}}
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.25,
            "top_p": 0.75,
            "num_predict": 220
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=AI_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()
        raw = data.get("response", "")

        return extract_json(raw)

    except Exception as e:
        print("REPAIR ERROR:", e)
        return None


# ============================================================
# SPECIAL SAFE EXPLANATIONS
# ============================================================

def get_safe_explanations(situation):
    """
    Deterministic fallback for common short-reply situations.

    This is not used to generate the entire experience. It is a
    safety fallback when the local 3B model produces invalid output.
    """

    facts = get_facts(situation)
    s = situation.lower()

    # Short "Okeyy" / "Okay" situation
    if facts["okay_reply"]:

        return [
            {
                "title": "Possible explanation",
                "reason": (
                    "She may simply have been acknowledging your message "
                    "without having anything else to add."
                )
            },
            {
                "title": "Possible explanation",
                "reason": (
                    "\"Okeyy\" could be her casual way of showing that "
                    "she understood what you said."
                )
            },
            {
                "title": "Possible explanation",
                "reason": (
                    "She might not have known what else to say, so she "
                    "kept the reply short."
                )
            },
            {
                "title": "Possible explanation",
                "reason": (
                    "The extra \"y\" may simply be part of her texting "
                    "style rather than a sign of a particular emotion."
                )
            },
            {
                "title": "Possible explanation",
                "reason": (
                    "She may have wanted to keep that particular reply "
                    "brief without necessarily meaning anything more."
                )
            }
        ]

    # Generic conservative fallback
    return [
        {
            "title": "Possible explanation",
            "reason": (
                "There may be more than one reasonable interpretation "
                "of what happened."
            )
        },
        {
            "title": "Possible explanation",
            "reason": (
                "The action described does not necessarily have one "
                "specific meaning."
            )
        },
        {
            "title": "Possible explanation",
            "reason": (
                "The situation may have an ordinary explanation that "
                "cannot be determined from the available information."
            )
        },
        {
            "title": "Possible explanation",
            "reason": (
                "The available details do not provide enough evidence "
                "to identify a definite reason."
            )
        },
        {
            "title": "Possible explanation",
            "reason": (
                "It is possible that the behavior was less meaningful "
                "than it initially seemed."
            )
        }
    ]


# ============================================================
# CLEAN AI RESULTS
# ============================================================

def process_results(situation, ai_data):
    facts = get_facts(situation)

    raw_results = []

    if isinstance(ai_data, dict):
        raw_results = ai_data.get("results", [])

    if not isinstance(raw_results, list):
        raw_results = []

    valid_results = []

    for item in raw_results:

        if not isinstance(item, dict):
            continue

        reason = clean_text(
            item.get("reason", "")
        )

        title = clean_text(
            item.get("title", "Possible explanation")
        )

        if validate_explanation(reason, situation, facts):

            valid_results.append({
                "title": title or "Possible explanation",
                "reason": reason
            })

    # If enough valid AI results exist, use them
    if len(valid_results) >= 5:
        valid_results = valid_results[:5]

    else:
        # Try repair
        repair_input = []

        for item in raw_results:
            if isinstance(item, dict):
                repair_input.append({
                    "title": item.get("title", ""),
                    "reason": item.get("reason", "")
                })

        repaired = repair_explanations(
            situation,
            repair_input
        )

        if repaired and isinstance(repaired, dict):

            repaired_results = repaired.get(
                "results",
                []
            )

            if isinstance(repaired_results, list):

                for item in repaired_results:

                    if not isinstance(item, dict):
                        continue

                    reason = clean_text(
                        item.get("reason", "")
                    )

                    title = clean_text(
                        item.get(
                            "title",
                            "Possible explanation"
                        )
                    )

                    if validate_explanation(
                        reason,
                        situation,
                        facts
                    ):
                        valid_results.append({
                            "title": title or "Possible explanation",
                            "reason": reason
                        })

                        if len(valid_results) >= 5:
                            break

    # Final deterministic fallback
    if len(valid_results) < 5:

        safe = get_safe_explanations(
            situation
        )

        for item in safe:

            duplicate = False

            for existing in valid_results:

                if existing["reason"].lower() == item["reason"].lower():
                    duplicate = True
                    break

            if not duplicate:
                valid_results.append(item)

            if len(valid_results) >= 5:
                break

    # Guarantee exactly five
    while len(valid_results) < 5:

        valid_results.append({
            "title": "Possible explanation",
            "reason": (
                "There is not enough information to know the exact "
                "reason from the situation alone."
            )
        })

    return valid_results[:5]


# ============================================================
# SCORE
# ============================================================

def calculate_score(situation, results):
    """
    Simple evidence-based overthinking score.

    This intentionally does not claim to be a scientific psychological
    measurement. It is a playful simulator score.
    """

    s = situation.lower()

    score = 40

    # More uncertainty = more overthinking
    uncertainty_words = [
        "why",
        "does this mean",
        "what does it mean",
        "did i do something",
        "is she mad",
        "is he mad",
        "does she hate",
        "does he hate",
        "ignoring me",
        "avoiding me",
        "weird",
        "strange",
        "only",
    ]

    for word in uncertainty_words:
        if word in s:
            score += 5

    # Very short social interactions often trigger overthinking
    if len(situation.split()) <= 12:
        score += 5

    # Multiple questions
    if situation.count("?") >= 2:
        score += 10

    # Strong emotional interpretation
    emotional_words = [
        "hate",
        "angry",
        "mad",
        "jealous",
        "ignored",
        "rejected",
        "embarrassed",
        "awkward",
    ]

    for word in emotional_words:
        if word in s:
            score += 3

    score = max(0, min(100, score))

    return score


def score_label(score):
    if score < 30:
        return "BARELY OVERTHINKING"

    if score < 50:
        return "SLIGHTLY OVERTHINKING"

    if score < 70:
        return "DEFINITELY OVERTHINKING"

    if score < 85:
        return "SERIOUSLY OVERTHINKING"

    return "PROFESSIONAL OVERTHINKER"


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    try:
        data = request.get_json(silent=True) or {}

        # IMPORTANT:
        # Frontend sends "situation".
        # Older versions used "question".
        # Supporting both prevents the 400 bug.
        question = (
            data.get("situation")
            or data.get("question")
            or data.get("text")
            or ""
        ).strip()

        if not question:
            return jsonify({
                "error": "Please enter something to overthink."
            }), 400

        # Prevent enormous prompts
        question = question[:1000]

        print("\n" + "=" * 60)
        print("NEW SITUATION:")
        print(question)
        print("=" * 60)

        # Ask local AI
        ai_data = ask_local_ai(question)

        # Process / validate / repair
        results = process_results(
            question,
            ai_data
        )

        # Score
        score = calculate_score(
            question,
            results
        )

        label = score_label(score)

        # Verdict
        facts = get_facts(question)

        verdict = ""

        if isinstance(ai_data, dict):
            verdict = clean_text(
                ai_data.get("verdict", "")
            )

        if not validate_verdict(
            verdict,
            question,
            facts
        ):
            verdict = ""

        # Try repaired verdict
        if not verdict:

            repaired = repair_explanations(
                question,
                results
            )

            if repaired and isinstance(repaired, dict):

                possible_verdict = clean_text(
                    repaired.get("verdict", "")
                )

                if validate_verdict(
                    possible_verdict,
                    question,
                    facts
                ):
                    verdict = possible_verdict

        # Final safe verdict
        if not verdict:
            verdict = safe_verdict(question)

        # Add UI data
        final_results = []

        for i, item in enumerate(results):

            final_results.append({
                "number": i + 1,
                "title": item["title"],
                "reason": item["reason"],
                "confidence": WEIGHTS[i],
                "weight": WEIGHTS[i]
            })

        response = {
            "score": score,
            "label": label,
            "results": final_results,
            "verdict": verdict
        }

        print("\nRESULT:")
        print(json.dumps(
            response,
            indent=2,
            ensure_ascii=False
        ))

        return jsonify(response)

    except Exception as e:

        print("\nANALYZE ERROR:")
        print(e)

        return jsonify({
            "error": (
                "Something went wrong while analyzing the situation."
            )
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    try:

        response = requests.get(
            "http://127.0.0.1:11434/api/tags",
            timeout=5
        )

        if response.ok:
            return jsonify({
                "status": "online",
                "model": MODEL
            })

    except Exception:
        pass

    return jsonify({
        "status": "offline",
        "model": MODEL
    }), 503


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("OVERTHINKING.AI")
    print("Local AI Overthinking Simulator")
    print("=" * 60)
    print("Model:", MODEL)
    print("Ollama:", OLLAMA_URL)
    print("Server: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )