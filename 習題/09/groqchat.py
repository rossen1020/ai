import os
from groq import Groq

#API key
os.environ["GROQ_API_KEY"] = "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"

# 與Groq連接並進行API驗證
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def chat():
    print("你好！請開始與我交流。結束對話請輸入 '掰掰' ")  #開始對話
    
    while True:
        user_input = input("你: ")
        
        if user_input.lower() == "掰掰":
            print("掰掰~")
            break
        
        #代入老師給的範例
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": user_input,
                    
                }
            ],
            model="llama3-8b-8192", #所使用的模型
        )
        
        # 模型給的回應內容
        response = chat_completion.choices[0].message.content
        print("groq:", response)


if __name__ == "__main__":
    chat()
