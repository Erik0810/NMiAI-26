"""Estimate API costs for gemini-3.1-pro-preview."""

# Estimate based on typical tool-calling agent token consumption
# Our agent: ~8K system prompt, ~15K tool definitions, growing conversation history
# Average task: 5 turns

turns = 5
avg_input = 35000  # input per turn (system + tools + history, grows each turn)
avg_output = 800
avg_thinking = 4000

total_input = turns * avg_input
total_output = turns * avg_output
total_thinking = turns * avg_thinking

print("=== Estimated tokens per task ===")
print(f"  Input:    {total_input:,}")
print(f"  Output:   {total_output:,}")
print(f"  Thinking: {total_thinking:,}")
print()

# Pricing tiers (Gemini 2.5 Pro as reference since 3.1 pricing not published yet)
scenarios = [
    ("Low (Flash-like)", 0.15, 0.60, 0.60),
    ("Mid (2.5-Pro rates)", 1.25, 10.0, 3.50),
    ("High (premium)", 2.50, 20.0, 7.00),
]

for label, inp_rate, out_rate, think_rate in scenarios:
    cost = (total_input/1e6 * inp_rate) + (total_output/1e6 * out_rate) + (total_thinking/1e6 * think_rate)
    print(f"{label}:")
    print(f"  Per task: ${cost:.3f}")
    print(f"  Our 4 tests: ${cost * 4:.2f}")
    print(f"  30 tasks: ${cost * 30:.2f}")
    print(f"  300 submissions: ${cost * 300:.2f}")
    print()

print("CHECK ACTUAL SPEND:")
print("  https://aistudio.google.com/apikey  -> click key -> View metrics")
print("  https://console.cloud.google.com/billing")
