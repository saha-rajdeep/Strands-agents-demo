# **Install Strands Agents SDK and tools** 
# pip install strands-agents strands-agents-tools
# **Install the required dependency**
# pip install 'strands-agents[anthropic]'

from strands import Agent
from strands.models.anthropic import AnthropicModel
from strands_tools import current_time, http_request, use_aws

# Initialize Claude model with your API key
  
model = AnthropicModel(
    client_args={
        "api_key": "<Your API Key>"  # Replace with your key OR comment the model line in agent to use Bedrock default
    },
    model_id="claude-3-7-sonnet-20250219",  # Use Claude 
    max_tokens=1000
)
WEATHER_SYSTEM_PROMPT = """You are a weather assistant with HTTP capabilities. You can:
1. Make HTTP requests to the National Weather Service API
2. Process and display weather forecast data
3. Provide weather information for locations in the United States

When displaying responses:
- Format weather data in a human-readable way
- Highlight important information like temperature, precipitation, and alerts
- Handle errors appropriately
- Convert technical terms to user-friendly language

Always explain the weather conditions clearly and provide context for the forecast.
"""  
# Create agent with Claude 4
agent = Agent(
    model=model,
    tools=[current_time, http_request,use_aws],
    system_prompt=WEATHER_SYSTEM_PROMPT
)

# Use the agent
#response = agent("Explain quantum computing in simple terms")
#print(response)

response = agent("1. what is the time in new york city? 2. What is the weather in new york city? 3. list the s3 buckets in my aws account")
print(response)










