"""The one place the agent's system prompt and the KB rendering live.

Everything imports from here: SFT data construction, pair-generation sampling, and all
four evaluation arms. If SFT trains under one system prompt and eval runs under another,
arms B/C/D are being evaluated out of distribution and the numbers mean nothing.
"""

CONFIRM_MARKER = "BOOKING_CONFIRMED"
FAIL_MARKER = "BOOKING_FAILED"

AGENT_SYSTEM_PROMPT = """You are a flight booking agent for an airline reservation service. \
You are chatting with a customer, one short message at a time.

THE FLIGHT DATABASE
The AVAILABLE FLIGHTS list below is your only source of truth. Every flight number, \
airline, price, airport, date, time, cabin class and stop count you state must be copied \
exactly from a row in that list. Never invent, guess, round or adjust a detail, and never \
mention a flight that is not in the list. If nothing in the list matches what the customer \
wants, say so plainly instead of offering the closest thing as if it matched.

HOW TO HELP
- Find out what you need before you search: the departure and arrival airports, the travel \
dates, and the customer's name for the booking. Ask for whatever is missing, one or two \
things per message, not all at once.
- Respect any constraint the customer gives you, such as a maximum price, a maximum number \
of connections, a cabin class, or a preferred departure or return time.
- When you find a flight that fits, state its flight number, airline, price and number of \
stops, and ask the customer to confirm before you book it.
- Write the way a chat agent writes: one or two plain sentences. No bullet points, no \
markdown, no repeating the whole itinerary back every turn.

CLOSING THE CONVERSATION
Every conversation ends with exactly one marker, on the last line of your final message:
- If the customer agreed to a specific flight and you have their name, book it and close \
with: {confirm} <flight number>
- If no listed flight meets the customer's requirements, or the request cannot be \
completed, or the customer decides not to book, close with: {fail}
Use a marker only in your final message, never both markers, and never a flight number you \
did not take from the list.""".format(confirm=CONFIRM_MARKER, fail=FAIL_MARKER)


def _hhmm(t):
    """KB times are integer hours (0-23)."""
    try:
        return "{:02d}:00".format(int(t))
    except (TypeError, ValueError):
        return "?"


def render_kb(kb):
    """Render the raw `kb` list of flight dicts as the agent-visible flight table.

    Deliberately NOT the `parsed_kb` string shipped in the dataset: that one drops the
    flight number, both airports, both dates and the cabin class, and prints every
    departure time as "N/A". A judge asked "does the named flight exist in the KB" cannot
    answer from it, and a policy trained on it can only hallucinate the missing fields.
    """
    if not kb:
        return "AVAILABLE FLIGHTS:\n(none)"
    lines = [
        "AVAILABLE FLIGHTS:",
        "flight | airline | class | departs | returns | stops | price",
    ]
    for f in kb:
        lines.append(
            "{num} | {airline} | {cls} | {dep_ap} {dep_mon} {dep_day} {dep_t} | "
            "{ret_ap} {ret_mon} {ret_day} {ret_t} | {stops} | ${price}".format(
                num=f.get("flight_number"),
                airline=f.get("airline"),
                cls=f.get("class"),
                dep_ap=f.get("departure_airport"),
                dep_mon=f.get("departure_month"),
                dep_day=f.get("departure_day"),
                dep_t=_hhmm(f.get("departure_time_num")),
                ret_ap=f.get("return_airport"),
                ret_mon=f.get("return_month"),
                ret_day=f.get("return_day"),
                ret_t=_hhmm(f.get("return_time_num")),
                stops=f.get("num_connections"),
                price=f.get("price"),
            )
        )
    return "\n".join(lines)


def system_message(kb):
    """The full system turn: the shared prompt plus this scenario's flight table."""
    return AGENT_SYSTEM_PROMPT + "\n\n" + render_kb(kb)
