# Daily Assistant MCP Server

A practical MCP (Model Context Protocol) server that provides useful everyday tools for personal
productivity. From calculating tips to converting timezones, this server offers a collection of
handy utilities you can use in your daily life.

## 🛠️ Available Tools

### 💰 Financial Tools

- **`calculate_tip`**: Calculate tip amount and total bill with customizable tip percentage
- **`split_bill`**: Split a bill among multiple people, with or without tip

### 🔐 Security Tools

- **`generate_password`**: Generate secure random passwords with customizable options

### ⏰ Time & Date Tools

- **`convert_timezone`**: Convert time between different timezones
- **`countdown_timer`**: Calculate time remaining until a target date/event

### 🌡️ Health & Weather Tools

- **`calculate_bmi`**: Calculate Body Mass Index and health category
- **`get_weather`**: Get current weather for any city (demo with mock data)

### 📏 Utility Tools

- **`unit_converter`**: Convert between common units (length, weight, temperature, volume)
- **`dice_roll`**: Roll dice with customizable number and sides
- **`text_statistics`**: Analyze text and provide word count, reading time, and more

## Prerequisites

- **Python**: 3.11+
- **Dependencies**: managed via `uv` (or use `pip` if you prefer)

## Install

```bash
uv sync
```

If you are using `pip` instead of `uv`, it is recommended to create a virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install "mcp>=1.12.4" "requests>=2.31.0" "python-dateutil>=2.8.2"
```

## Run the server locally

```bash
uv run main.py
```

The MCP endpoint will be available at:

- `http://localhost:8080/mcp`

## Example Usage

Here are some example uses of the available tools. You can run `client.py` to test the tools
programmatically.

### Calculate a tip

```python
# Calculate 20% tip on a $45.50 bill
calculate_tip(45.50, 20)
# Returns: {"original_amount": 45.50, "tip_percentage": 20, "tip_amount": 9.10, "total_amount": 54.60}
```

### Split a bill

```python
# Split $120 among 4 people with 18% tip
split_bill(120, 4, include_tip=True, tip_percentage=18)
# Returns breakdown per person including tip
```

### Generate a password

```python
# Generate a 20-character password without ambiguous characters
generate_password(20, include_symbols=True, exclude_ambiguous=True)
# Returns: "Kd3#mN9$pQ7@xR5&vW2!"
```

### Convert timezone

```python
# Convert current time from UTC to Tokyo time
convert_timezone("now", "UTC", "Asia/Tokyo")
# Returns time in both timezones with difference
```

### Calculate BMI

```python
# Calculate BMI for 70kg, 175cm
calculate_bmi(70, 175)
# Returns: {"bmi": 22.9, "category": "Normal weight", ...}
```

### Convert units

```python
# Convert 5 kilometers to miles
unit_converter(5, "km", "miles")
# Returns: {"original_value": 5, "converted_value": 3.1069, ...}
```

## 🔗 MCP Client Integration

This MCP can be integrated with AI assistants like Cursor or Claude Desktop.

### Setting up with Cursor

1. **Add to your `.cursor/mcp.json` file:**

```json
{
  "mcpServers": {
    "Daily Assistant MCP (Run locally)": {
      "url": "http://localhost:8080/mcp"
    },
    "Daily Assistant MCP (Deployed on Friendli Apps)": {
      "url": "https://api.friendli.ai/inference/<app-id>/mcp"
    }
  }
}
```

2. **Restart Cursor** to load the new MCP server

## 💬 Real Conversation Examples

Here are examples of how you can naturally interact with the tools through an AI assistant:

### 💰 Restaurant Bill Scenario

**You:** "I'm at dinner with 3 friends. Our bill is $127.80. How much should each person pay with a
20% tip?"

**AI:** *I'll help you split that bill with tip included.*

```
🧮 Calculating your split bill...
• Original amount: $127.80
• 20% tip: $25.56
• Total with tip: $153.36
• Split 4 ways: $38.34 per person
```

### 🌍 Travel Planning Scenario

**You:** "I have a video call at 3 PM EST tomorrow. What time is that in Tokyo? Also, if the
temperature there is 22°C, what's that in Fahrenheit?"

**AI:** *Let me convert both the time and temperature for you.*

```
⏰ Time Conversion:
• 3:00 PM EST = 5:00 AM JST (next day)
• Time difference: +14 hours

🌡️ Temperature Conversion:
• 22°C = 71.6°F
```

### 🏃‍♀️ Health & Fitness Scenario

**You:** "I'm 5'9" and weigh 154 pounds. What's my BMI? Also, I want to track a 5K run - how many
miles is that?"

**AI:** *I'll calculate your BMI and convert the distance for you.*

```
📊 Health Stats:
• Height: 175 cm, Weight: 70 kg
• BMI: 22.9 (Normal weight)

🏃‍♀️ Distance Conversion:
• 5 kilometers = 3.1 miles
```

## Why These Tools?

This MCP server focuses on practical, everyday utilities that people actually need:

- **No complex setup**: All tools work out of the box without API keys or configuration
- **Universally useful**: Tools that apply to common daily situations
- **Clear value**: Each tool solves a specific, relatable problem
- **Easy to understand**: Simple inputs and outputs that make sense immediately

## Project Structure

```text
daily-assistant-mcp/
├── main.py          # MCP server with all the tool implementations
├── client.py        # Test client for the tools in daily-assistant-mcp
├── pyproject.toml   # Project configuration and dependencies
└── README.md        # This file
```

## 🔄 Troubleshooting

### Common Issues

**Server won't start:**

- Ensure Python 3.11+ is installed
- Check if port 8080 is available: `lsof -i :8080`
- Verify dependencies: `uv sync`

**MCP client can't connect:**

- Confirm server is running: `curl http://localhost:8080/mcp`
- Check firewall settings
- Restart your MCP client after configuration changes

**Tools not appearing:**

- Check server logs for errors
- Ensure correct URL format

## Future Enhancements

Potential additions for even more utility:

- Recipe scaling calculator
- Loan/mortgage calculator
- Color code converter (HEX/RGB/HSL)
- QR code generator
- URL shortener
- Reminder/timer tool
- Currency converter with live rates
- Base64 encoder/decoder
- Hash calculator (MD5, SHA256)
- JSON formatter/validator
