# power-cosumption-chatbot

Name: Parth Jain

A chatbot created using LangGraph to answer user's questions by generating and safely running pandas queries on 
[Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) dataset.

## LLMs used:
1. Groq - meta-llama/llama-4-maverick-17b-128e-instruct
2. Gemini - gemini-2.0-flash

## Summary
- [X] Generate and execute pandas queries.
- [X] Create graphs such as bar graphs or pie charts from the given data.
- [ ] Integrate with frontend

### Data collection and pre-processing
```py
import pandas as pd

df = pd.read_csv('household_power_consumption.txt',
                sep=';',
                na_values=['?'],
                low_memory=False)

df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df = df.drop(['Date', 'Time'], axis=1)

df = df.dropna()
df['Global_active_power'] = df['Global_active_power'].astype(float)

df.columns = df.columns.str.lower()
```

### LLM choice
The user can select either Gemini or Groq by setting the `LLM_PROVIDER` variable to `groq` or `gemini`. 

For the selected provider, add `GROQ_API_KEY` or `GOOGLE_API_KEY` as a secret in colab, or comment the part and provide the API keys.

### Custom Pandas Tool
Created a custom LangChain tool using a function `create_and_execute_pandas_query()` that generates and runs pandas query based on the given input.

It can also generate plots using matplotlib.pyplot if required.

Prompt used:
```py
prompt = f"""
You are a helpful assistant that converts natural language questions into **one-line executable Pandas code**.
  Use the DataFrame `df`. Only use pandas operations — no imports, no built-ins.

  Guidelines:
  - Think before coding. Ensure the output is correct and efficient.
  - Try to find summary type data like count, average etc. instead of returning rows upon rows of the df.
  - NEVER return more than 10-20 rows of the df.
  - If the question can't be answered via the DataFrame, return a single Python comment explaining the logic.
  - `global_active_power` is in **kilowatts (kW)** and sampled **every minute**.
    - This represents instantaneous power, not energy.
    - If a question uses terms like "energy", "consumption", or "usage", interpret it as **total energy in kilowatt-hours (kWh)**.
        - Compute this by resampling (e.g., daily) and using: `(global_active_power.sum() / 60)`
    - Only use `.mean()` if the question clearly asks for "average power", not "energy" or "usage".
  - Aggregate before filtering. For summaries over days/months/years, use `groupby(df['datetime'].dt.<unit>)` or `resample('<unit>', on='datetime')`.
  - Do not use per-minute data for daily or higher-level summaries.
  - For potentially large outputs, return only `.head(10)` or a summary.
  - Assign your result to the variable `output`.
  - You may also generate matplotlib plots (e.g., df.plot(...)) and assign the plot object to `output`.
  - Do NOT call `plt.show()`, just assign the plot to `output = df.plot(...)` or similar.


  The DataFrame has these columns:
  - `datetime`: datetime64[ns] (%d/%m/%Y %H:%M:%S)
  - `global_active_power`: float (kW)
  - `global_reactive_power`: float
  - `voltage`: float
  - `global_intensity`: float
  - `sub_metering_1`: int (Wh)
  - `sub_metering_2`: int (Wh)
  - `sub_metering_3`: int (Wh)

  Examples:

  Q: What is the average global active power in March 2007?
  A: result = df[df['datetime'].dt.month == 3 & (df['datetime'].dt.year == 2007)]['global_active_power'].mean()


  Question: {question}
"""
```

The LLM returns a response as:
```py
class PandasReponseFormat(BaseModel):
  thoughts: str = Field(description="The thoughts before writing the pandas query")
  code: str = Field(description="The required pandas query")
```

### Initiating the agent
A langGraph based reAct agent that uses the provided tools to reason with itself and generate the answer.

```py
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(llm, [create_and_execute_pandas_query], checkpointer=memory)
```

System prompt for the agent:
```py
system_message = {
    "role": "user",
    "content": """You are a data analysis assistant. You answer questions about electricity consumption from Individual Household Electric Power Consumption.
It contains measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years.

You should verify the generated code, if you find something wrong, you may use the tool again by also providing instructions to fix the issue.
If the tool responded that it added a graph, let the user know where they can find the plot.
Your role is to use the provided tool to satisfy user's query.

Ensure that you include the complete answer to the user's question in the final output, it should answer everything without any details from the previous chains.
If the results are ambiguous, try a different approach for the tool.
"""
}
```


### REPL chat interface

A simple inteface to interact with the agent.

```py
while True:
  user_input = input("Chat: ")

  if user_input.lower() == 'quit':
    print("Goodbye!")
    break

  input_message = user_input
  for event in agent_executor.stream(
    {"messages": [system_message, {"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
  ):
    event["messages"][-1].pretty_print()
```





