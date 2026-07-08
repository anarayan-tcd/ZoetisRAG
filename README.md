# ZoetisRAG
Code for RAG/Agentic python script for Zoetis in collab with Trinity Business School

##Main Files
Knowledge_agent.py
>This code allows out LLM Agents using Gemini Flash 2.5 to parse the internet and collect official medication details to store in our database.
>This is a low-execution file, and ideally should only be executed once every few weeks, or if any new products are launched.
>This data is stored in a local ChromaDB database, but can and should be replaced by a more robust SQL DB. Currently, the new data is overwritten each time, but this will be replaced to only add/modify data.

Query_agent.py
>This code block runs a cheaper/lighter LLM, Deepseek in this case, to take the input list of strings (Vet Prescriptions) and reason on them to match them to the closest actual medicine variant.
>it uses the chromaDB text to generate context for each reasoning task.
>outputs in JSON

Gemini_app.py
>this is the app interface, which uses Query_agent.py's output to display it in an app-like interface.
