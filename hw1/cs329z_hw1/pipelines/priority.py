import json
import re

from cs329z_hw1.types import Email, LMCallable

CATEGORIES = {"urgent", "normal", "ignore"}

SYSTEM_PROMPT = """
    You classify one email's priority from the point of view of the recipient
    (the people in the To field), at the time it was sent.

    Choose exactly one category:

    - urgent: the recipient must act or decide soon. BOTH conditions must hold:
    (1) the email asks the recipient to do something, or reports a live problem
        the recipient is responsible for; and
    (2) action is needed the same day, the next working day, or by a stated
        deadline no more than two working days away.
    - normal: legitimate work or personal mail written for this recipient or a
    small working group, with no deadline in the next two working days.
    - ignore: advertising, newsletters, mass announcements to large lists, and
    automated notifications that ask nothing of the recipient.

    Important:
    - Judge from the recipient's perspective, not by how important the topic is
    to the company.
    - Do not decide by wording alone. A mass mailing saying "act now" is ignore.
    An email with "URGENT" in the subject that asks nothing of the recipient
    is not urgent.
    - An automated notice that asks this specific recipient to act is normal, or
    urgent if its deadline is within two working days.

    Return exactly one JSON object:
    {"category": "urgent" | "normal" | "ignore", "reason": "one concise sentence"}
    Do not add any other text outside the JSON object.
"""

def classify_priority(email: Email, lm: LMCallable) -> dict:
    # Message = dict  # {"role": "system" | "user" | "assistant", "content": str}
    recipients = ",".join(email.get("to") or [])
    user_prompt = (
        f"From: {email.get('from', '')}\n"
        f"To: {recipients}\n"
        f"Date: {email.get('date', '')}\n"
        f"Subject: {email.get('subject', '')}\n"
        f"Body: {email.get('body', '')}\n"
    )

    try:
        reply = lm([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as exc:
        return {
            "category": "unknown", 
            "reason": f"The model call failed : {exc}",
        }
    text = reply.strip() if isinstance(reply, str) else ""

    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        data = None

    if isinstance(data, dict):
        category = data.get("category")
        if isinstance(category, str) and category.strip().lower() in CATEGORIES:
            category = category.strip().lower()
            reason = data.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                reason = f"The model classified the email as {category}."
            return {"category": category, "reason": reason.strip()}

        return {"category": "unknown", "reason": "The model reply did not state a valid priority label.",}

    explicit = re.search(r"\b(?:category|label)\s*[=:]\s*[\"']?(urgent|normal|ignore)\b",
        text,
        flags=re.IGNORECASE,
    )
    if explicit:
        category = explicit.group(1).lower()
        return {
            "category": category, "reason": text or f"The model classified the email as {category}.",
        }

    matches = re.findall(
        r"\b(urgent|normal|ignore)\b",
        text,
        flags=re.IGNORECASE,
    )
    distinct = {match.lower() for match in matches}

    if len(distinct) == 1:
        category = next(iter(distinct))
        return {
            "category": category,
            "reason": text or f"The model classified the email as {category}.",
        }

    return {
        "category": "unknown",
        "reason": "The model reply did not state a single valid priority label.",
    }