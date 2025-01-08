import time
import os
import json


def get_response(prompts,client,model_name="gpt-4o-2024-08-06",max_tokens=4096,temperature=0.8,system=None):
    answer_list=[]
    for prompt in prompts:
        try:
            messages =[{"role": "user", "content": prompt}]
            if system is not None:
                messages.insert(0,{"role": "system", "content": system})

            if "gemini" not in model_name.lower():
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=False,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    timeout=60,
                )
                answer=response.choices[0].message.content
            else:
                model = client.GenerativeModel(model_name=model_name,system_instruction=system)
                response = model.generate_content(prompt,
                                                  generation_config={"max_output_tokens": max_tokens, "temperature": temperature,
                                                                     "top_p": 0.9,})
                answer=response.text
                time.sleep(10)
        except Exception as e:
            print(model_name,e)
            time.sleep(10)
            answer=""

        answer_list.append(answer)

    return answer_list

def get_client(test_model_name,base_url=None):
    from openai import OpenAI
    if "gpt-4o" in test_model_name:
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        client = OpenAI(api_key="your api key", base_url="https://api.chatanywhere.tech/v1/")
    elif "deepseek" in test_model_name:
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        client=  OpenAI(api_key="your api key", base_url="https://api.deepseek.com/beta")
    elif "gemini" in test_model_name:
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

        api_key="your api key"
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        client=genai
    else:
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        client = OpenAI(api_key="EMPTY", base_url="http://localhost:5001/v1" if base_url is None else base_url)

    return client

def get_azure_client():
    pass

def model_answer(prompt,model_name,max_tokens=4096*2,temperature=0.8,base_url=None,azure=False,system=None):
    if azure:
        client=get_azure_client()
    else:
        client=get_client(model_name,base_url=base_url)

    if isinstance(prompt,str):
        prompt=[prompt]
        response = get_response(prompt, client, model_name=model_name, max_tokens=max_tokens, temperature=temperature,
                                system=system)
        return response[0]
    else:
        response=get_response(prompt,client,model_name=model_name,max_tokens=max_tokens,temperature=temperature,system=system)
        return response


# gemini 12.5 $/1000k token
# gpt4o input: 5$/1M token  output: 15$/1M token

if __name__ == '__main__':
    # prompt="你是谁？"
    # model_name="gpt-4o-mini"
    # response=model_answer(prompt,model_name,azure=False)
    # print(response)

    prompt="你是谁？"
    model_name="mistral-v0.1-chinese"
    response=model_answer(prompt,model_name,azure=False,base_url="http://localhost:5006/v1",max_tokens=1000)
    print(response)