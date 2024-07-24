from custom_chat_langchain import APIBasedChatLLM

llm = APIBasedChatLLM(stream = True)

input_content = "tell me a poem in two paragraphs"
# print(llm.invoke(input_content).content)
for text in llm.stream(input=input_content):
    print(text.content,end="")