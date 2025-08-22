"""Daily Assistant MCP Server - Practical tools for everyday productivity."""

import random
import re
from datetime import datetime
from typing import Any

from dateutil import parser, tz
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(host="0.0.0.0", stateless_http=True, port=8080)


@mcp.tool()
def calculate_tip(amount: float, tip_percentage: float = 15.0) -> dict[str, float]:
    """Calculate tip amount and total bill.

    Args:
        amount: The bill amount before tip
        tip_percentage: Tip percentage (default 15%)

    Returns:
        Dictionary with tip amount and total bill
    """
    tip = amount * (tip_percentage / 100)
    total = amount + tip
    return {
        "original_amount": round(amount, 2),
        "tip_percentage": tip_percentage,
        "tip_amount": round(tip, 2),
        "total_amount": round(total, 2),
    }


@mcp.tool()
def split_bill(
    total_amount: float, num_people: int, include_tip: bool = True, tip_percentage: float = 15.0
) -> dict[str, Any]:
    """Split a bill among multiple people.

    Args:
        total_amount: The total bill amount
        num_people: Number of people to split between
        include_tip: Whether to add tip before splitting
        tip_percentage: Tip percentage if including tip

    Returns:
        Breakdown of amounts per person
    """
    if include_tip:
        tip = total_amount * (tip_percentage / 100)
        final_total = total_amount + tip
    else:
        tip = 0
        final_total = total_amount

    per_person = final_total / num_people

    return {
        "original_amount": round(total_amount, 2),
        "tip_amount": round(tip, 2),
        "final_total": round(final_total, 2),
        "num_people": num_people,
        "per_person": round(per_person, 2),
    }


@mcp.tool()
def generate_password(
    length: int = 16, include_symbols: bool = True, exclude_ambiguous: bool = True
) -> str:
    """Generate a secure random password.

    Args:
        length: Password length (default 16)
        include_symbols: Include special characters
        exclude_ambiguous: Exclude ambiguous characters like 0,O,l,1

    Returns:
        Generated password
    """
    lowercase = "abcdefghijklmnopqrstuvwxyz"
    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits = "0123456789"
    symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?"

    if exclude_ambiguous:
        lowercase = lowercase.replace("l", "")
        uppercase = uppercase.replace("O", "").replace("I", "")
        digits = digits.replace("0", "").replace("1", "")
        symbols = symbols.replace("|", "").replace("l", "")

    chars = lowercase + uppercase + digits
    if include_symbols:
        chars += symbols

    # Ensure at least one character from each category
    password = [random.choice(lowercase), random.choice(uppercase), random.choice(digits)]
    if include_symbols:
        password.append(random.choice(symbols))

    # Fill the rest randomly
    for _ in range(length - len(password)):
        password.append(random.choice(chars))

    # Shuffle the password
    random.shuffle(password)
    return "".join(password)


@mcp.tool()
def convert_timezone(
    time_str: str, from_timezone: str = "UTC", to_timezone: str = "America/New_York"
) -> dict[str, str]:
    """Convert time between timezones.

    Args:
        time_str: Time string to convert (e.g., "2024-01-15 14:30" or "now")
        from_timezone: Source timezone (default UTC)
        to_timezone: Target timezone (default America/New_York)

    Returns:
        Converted time information
    """
    if time_str.lower() == "now":
        dt = datetime.now(tz.UTC)
        from_timezone = "UTC"
    else:
        # Parse the datetime string
        dt = parser.parse(time_str)

        # If no timezone info, assume it's in from_timezone
        if dt.tzinfo is None:
            from_tz = tz.gettz(from_timezone)
            dt = dt.replace(tzinfo=from_tz)

    # Convert to target timezone
    to_tz = tz.gettz(to_timezone)
    converted_dt = dt.astimezone(to_tz)

    return {
        "original_time": dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "original_timezone": from_timezone,
        "converted_time": converted_dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "converted_timezone": to_timezone,
        "time_difference": str(converted_dt.utcoffset()),
    }


@mcp.tool()
def calculate_bmi(weight_kg: float, height_cm: float) -> dict[str, Any]:
    """Calculate Body Mass Index (BMI) and health category.

    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters

    Returns:
        BMI value and health category
    """
    height_m = height_cm / 100
    bmi = weight_kg / (height_m**2)

    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 25:
        category = "Normal weight"
    elif 25 <= bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return {
        "bmi": round(bmi, 1),
        "category": category,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
    }


@mcp.tool()
def get_weather(city: str, units: str = "celsius") -> dict[str, Any]:
    """Get current weather for a city (demo with mock data).

    Args:
        city: City name
        units: Temperature units (celsius or fahrenheit)

    Returns:
        Weather information
    """
    # This is a mock implementation. In a real scenario, you'd use a weather API
    # For demonstration purposes, we'll return randomized but realistic data

    weather_conditions = ["Clear", "Partly Cloudy", "Cloudy", "Light Rain", "Rain", "Snow", "Fog"]
    condition = random.choice(weather_conditions)

    # Generate temperature based on condition
    if condition in ["Snow"]:
        temp_c = random.randint(-10, 5)
    elif condition in ["Rain", "Light Rain"]:
        temp_c = random.randint(5, 20)
    else:
        temp_c = random.randint(10, 30)

    temp_f = (temp_c * 9 / 5) + 32

    return {
        "city": city,
        "condition": condition,
        "temperature": temp_f if units == "fahrenheit" else temp_c,
        "units": "°F" if units == "fahrenheit" else "°C",
        "humidity": random.randint(30, 90),
        "wind_speed": random.randint(5, 30),
        "feels_like": (temp_f + random.randint(-5, 5))
        if units == "fahrenheit"
        else (temp_c + random.randint(-3, 3)),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@mcp.tool()
def countdown_timer(target_date: str, event_name: str | None = None) -> dict[str, Any]:
    """Calculate time remaining until a target date.

    Args:
        target_date: Target date string (e.g., "2024-12-25" or "2024-12-25 15:30")
        event_name: Optional name for the event

    Returns:
        Time remaining breakdown
    """
    target = parser.parse(target_date)
    if target.tzinfo is None:
        # Assume target is in local timezone
        target = target.replace(tzinfo=tz.tzlocal())

    now = datetime.now(tz.tzlocal())

    if target < now:
        time_diff = now - target
        is_past = True
    else:
        time_diff = target - now
        is_past = False

    days = time_diff.days
    hours, remainder = divmod(time_diff.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return {
        "event_name": event_name or "Target date",
        "target_date": target.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "current_date": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "is_past": is_past,
        "days": days,
        "hours": hours,
        "minutes": minutes,
        "seconds": seconds,
        "total_hours": days * 24 + hours,
        "total_minutes": (days * 24 + hours) * 60 + minutes,
        "human_readable": f"{days} days, {hours} hours, {minutes} minutes"
        + (" ago" if is_past else " remaining"),
    }


@mcp.tool()
def unit_converter(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    """Convert between common units.

    Args:
        value: The value to convert
        from_unit: Source unit (e.g., 'km', 'miles', 'kg', 'lbs', 'celsius', 'fahrenheit')
        to_unit: Target unit

    Returns:
        Converted value and details
    """
    conversions = {
        # Length
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("m", "ft"): 3.28084,
        ("ft", "m"): 0.3048,
        ("cm", "inch"): 0.393701,
        ("inch", "cm"): 2.54,
        # Weight
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
        ("g", "oz"): 0.035274,
        ("oz", "g"): 28.3495,
        # Temperature (special handling)
        ("celsius", "fahrenheit"): lambda c: (c * 9 / 5) + 32,
        ("fahrenheit", "celsius"): lambda f: (f - 32) * 5 / 9,
        # Volume
        ("liters", "gallons"): 0.264172,
        ("gallons", "liters"): 3.78541,
        ("ml", "oz"): 0.033814,
        ("oz", "ml"): 29.5735,
    }

    from_unit_lower = from_unit.lower()
    to_unit_lower = to_unit.lower()

    # Direct conversion
    if (from_unit_lower, to_unit_lower) in conversions:
        conversion = conversions[(from_unit_lower, to_unit_lower)]
        result = conversion(value) if callable(conversion) else value * conversion
    # Reverse conversion
    elif (to_unit_lower, from_unit_lower) in conversions:
        conversion = conversions[(to_unit_lower, from_unit_lower)]
        if callable(conversion):
            # For temperature, we need to apply the inverse function
            result = value * 9 / 5 + 32 if from_unit_lower == "celsius" else (value - 32) * 5 / 9
        else:
            result = value / conversion
    else:
        return {
            "error": f"Conversion from {from_unit} to {to_unit} not supported",
            "supported_units": [
                "km",
                "miles",
                "m",
                "ft",
                "cm",
                "inch",
                "kg",
                "lbs",
                "g",
                "oz",
                "celsius",
                "fahrenheit",
                "liters",
                "gallons",
                "ml",
                "oz",
            ],
        }

    return {
        "original_value": value,
        "original_unit": from_unit,
        "converted_value": round(result, 4),
        "converted_unit": to_unit,
        "conversion_rate": f"1 {from_unit} = {round(result / value, 6)} {to_unit}"
        if value != 0
        else "N/A",
    }


@mcp.tool()
def dice_roll(num_dice: int = 1, sides: int = 6) -> dict[str, Any]:
    """Roll dice and get results.

    Args:
        num_dice: Number of dice to roll (default 1)
        sides: Number of sides per die (default 6)

    Returns:
        Dice roll results
    """
    if num_dice < 1 or num_dice > 100:
        return {"error": "Number of dice must be between 1 and 100"}
    if sides < 2 or sides > 100:
        return {"error": "Number of sides must be between 2 and 100"}

    rolls = [random.randint(1, sides) for _ in range(num_dice)]

    return {
        "num_dice": num_dice,
        "sides": sides,
        "rolls": rolls,
        "sum": sum(rolls),
        "average": round(sum(rolls) / len(rolls), 2),
        "min": min(rolls),
        "max": max(rolls),
    }


@mcp.tool()
def text_statistics(text: str) -> dict[str, Any]:
    """Analyze text and provide statistics.

    Args:
        text: Text to analyze

    Returns:
        Various text statistics
    """
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Character counts
    total_chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))

    # Word analysis
    word_lengths = [len(word.strip('.,!?;:"')) for word in words]
    avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0

    # Readability estimate (simple)
    avg_words_per_sentence = len(words) / len(sentences) if sentences else 0

    return {
        "total_characters": total_chars,
        "characters_no_spaces": chars_no_spaces,
        "total_words": len(words),
        "total_sentences": len(sentences),
        "average_word_length": round(avg_word_length, 1),
        "average_words_per_sentence": round(avg_words_per_sentence, 1),
        "longest_word": max(words, key=len) if words else "",
        # Assuming 200 words per minute
        "reading_time_minutes": round(len(words) / 200, 1),
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
