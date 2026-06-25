from langchain_openai import ChatOpenAI  
import os  
import httpx  
client = httpx.Client(verify=False) 
llm = ChatOpenAI( 
base_url="https://genailab.tcs.in" ,
model = "azure/genailab-maas-gpt-4o-mini", 
api_key="sk-Zu-u-tb0bSC9YDwbj965TQ", 
http_client = client 
) 
llm.invoke("Hi")
print("hi")