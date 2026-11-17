from concurrent.futures import ThreadPoolExecutor

from cs329z_hw1.types import Email, LMCallable
from cs329z_hw1.pipelines.priority import classify_priority

MAX_WORKERS = 8
MAX_DIGEST_WORDS = 240



def daily_digest(emails: list[Email], lm: LMCallable) -> str:
    if not emails:
        return "No emails today."

    # classify concurrent
    def classify_one(email: Email)-> str:
        result = classify_priority(email, lm)
        category = result.get("category", "unknown")
        if category not in {"urgent", "normal", "ignore"}:
            category = "normal"

        return category

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        labels = list(pool.map(classify_one, emails))

    by_thread: dict[str, tuple[Email, str]] = {}

    for email, category in zip(emails, labels):
        if category == "ignore":
            continue
        thread_id = email.get("thread_id") or email["id"]
        old = by_thread.get(thread_id)

        if old is None or (old[1] != "urgent" and category == "urgent"):
            by_thread[thread_id] = (email, category)

    items = list(by_thread.values())

    urgent_items = [(email, cat) for email, cat in items if cat == "urgent"]
    normal_items = [(email, cat) for email, cat in items if cat == "normal"]

    if not urgent_items and not normal_items:
        return "No important emails today."

    def render(email: Email, full_body: bool) -> str:
        body = (email.get("body") or "").strip()

        if not full_body and len(body) > 500:
            body = body[:500] + "..."

        return (
            f"From: {email.get('from', '')}\n"
            f"Subject: {email.get('subject', '')}\n"
            f"Body: {body}\n"
        ) 

    urgent_text = "\n\n".join(
        render(email, full_body=True) for email, _ in urgent_items
    )
    normal_text = "\n\n".join(
        render(email, full_body=False) for email, _ in normal_items
    )

    system_prompt = f"""\
    You write a concise morning email digest for a busy employee.

    Rules:
    - Use only the emails provided.
    - The digest must be at most {MAX_DIGEST_WORDS} words.
    - Cover every urgent email.
    - For each urgent email, include the exact company, project, or site name
    from that email.
    - You may summarize normal emails briefly.
    - Do not mention advertising, newsletters, or mass announcements.
    - Do not invent facts.
    - Return only the digest text, with no preamble.
    """

    user_prompt = f"""\
    Write the morning digest.

    URGENT EMAILS:
    {urgent_text or "(none)"}

    NORMAL EMAILS:
    {normal_text or "(none)"}
    """

    reply = lm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    if not isinstance(reply, str):
        return "No important emails today."

    return reply.strip() or "No important emails today."