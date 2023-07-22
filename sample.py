# -*- coding: utf-8 -*-
import asyncio
import random
import blivedm
from tkinter import *
import pygame
import openai
import logging as log
import configparser
import time

from gtts import gTTS
import subprocess
from collections import deque
import requests
import pickle

config = configparser.ConfigParser()
#config.read('config.txt')
config.read('config.txt', encoding='utf-8-sig')


print(config.sections())
room_id = config.getint('DEFAULT', 'room_id')
openai_api_key = config.get('DEFAULT', 'openai.api_key')
set = config.get('DEFAULT', 'set')  # prompt
UID = config.get('DEFAULT', 'uid')  # 如果不设自己的uid，有时候会获取不到弹幕，而且必须传int
#打印set的类型
print(type(set))

print(room_id)
print(set)
MEMORYLEN = 4
print('memory len:',MEMORYLEN)
USE_MODEL = "OPENAI"#"chatglm-6b"  # chatglm 公司内网
print(f"using model {USE_MODEL}")


# 直播间ID的取值看直播间URL
TEST_ROOM_IDS = [
    room_id,#修改为你的直播间ID
]

async def main():
    #await run_single_client()
    await run_multi_client()
    await asyncio.sleep(20)

async def run_single_client():
    """
    演示监听一个直播间
    """
    room_id = random.choice(TEST_ROOM_IDS)
    # 如果SSL验证失败就把ssl设为False，B站真的有过忘续证书的情况
    client = blivedm.BLiveClient(room_id, ssl=True)
    handler = MyHandler()
    client.add_handler(handler)

    client.start()
    try:
        # 演示5秒后停止
        await asyncio.sleep(5)
        client.stop()

        await client.join()
    finally:
        await client.stop_and_close()


async def run_multi_client():
    """
    演示同时监听多个直播间
    """
    clients = [blivedm.BLiveClient(room_id, ssl=True, heartbeat_interval=30, uid=int(UID)) for room_id in TEST_ROOM_IDS]
    handler = MyHandler()
    for client in clients:
        client.add_handler(handler)
        client.start()

    try:
        await asyncio.gather(*(
            client.join() for client in clients
        ))
    finally:
        await asyncio.gather(*(
            client.stop_and_close() for client in clients
        ))


class MyHandler(blivedm.BaseHandler):
    # # 演示如何添加自定义回调
    # _CMD_CALLBACK_DICT = blivedm.BaseHandler._CMD_CALLBACK_DICT.copy()
    #
    # # 入场消息回调
    # async def __interact_word_callback(self, client: blivedm.BLiveClient, command: dict):
    #     print(f"[{client.room_id}] INTERACT_WORD: self_type={type(self).__name__}, room_id={client.room_id},"
    #           f" uname={command['data']['uname']}")
    # _CMD_CALLBACK_DICT['INTERACT_WORD'] = __interact_word_callback  # noqa
    def __init__(self):
        super().__init__()
        self.history = []
        

    async def _on_heartbeat(self, client: blivedm.BLiveClient, message: blivedm.HeartbeatMessage):
        print(f'[{client.room_id}] 当前人气值：{message.popularity}')

    #async def _on_danmaku(self, client: blivedm.BLiveClient, message: blivedm.DanmakuMessage):
    #    print(f'[{client.room_id}] {message.uname}：{message.msg}')

    
    #async def _on_gift(self, client: blivedm.BLiveClient, message: blivedm.GiftMessage):
    #    print(f'[{client.room_id}] {message.uname} 赠送{message.gift_name}x{message.num}'
    #          f' （{message.coin_type}瓜子x{message.total_coin}）')
    
    #async def _on_buy_guard(self, client: blivedm.BLiveClient, message: blivedm.GuardBuyMessage):
    #    print(f'[{client.room_id}] {message.username} 购买{message.gift_name}')

    #async def _on_super_chat(self, client: blivedm.BLiveClient, message: blivedm.SuperChatMessage):
    #   print(f'[{client.room_id}] 醒目留言 ¥{message.price} {message.uname}：{message.message}')

#回复弹幕
    def append_question(self, question):
        if USE_MODEL == "OPENAI":
            self.history.append({"role": "user", "content": question})
        elif USE_MODEL == "chatglm-6b":
            self.history.append({"question":question})

    def append_answer(self, answer):
        if USE_MODEL == "OPENAI":
            self.history.append({"role":"assistant","content":answer})
        elif USE_MODEL == "chatglm-6b" or USE_MODEL == 'chatglm2-6b':
            self.history[len(self.history)-1]["answer"] = answer

    def prepare_input(self, new_question):
        if USE_MODEL == "OPENAI":
            self.append_question(new_question)
            messages = [{"role": "system", "content": set}] + self.history
            return messages
        elif USE_MODEL == "chatglm-6b":
            prompt = set
            for i, text in enumerate(self.history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, text["question"], text["answer"])
            prompt += "[Round {}]\n问：{}\n答：".format(len(self.history), new_question)
            self.append_question(new_question)
            return prompt

    def clip_history(self):
        if USE_MODEL == 'OPENAI':
            if len(self.history) / 2 > MEMORYLEN:
                self.history.pop(0)
                self.history.pop(0)
        else:
            if len(self.history) > MEMORYLEN:
                self.history.pop(0)

    def send_openai(self, messages):
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        answer = str(response['choices'][0]['message']['content'])
        return answer

    def send_chatglm(self, text):
        url = 'http://alimama-llm.alibaba-inc.com/api'
        model_name = "chatglm-6b"
        headers = {   "Content-Type": "application/json; charset=UTF-8"}
        gen_config = {'max_new_tokens':40}
        data = {"model_name":model_name, "mode": 'generate',
          "data":{"text":text, "gen_config":gen_config}}
        input_len = len(text)
        try:
            r = requests.post(url, data=pickle.dumps(data), headers=headers)
            if r.status_code != 200:
                print(r.text)
                return "网络错误"
            result = r.json()['result']
            result = result[input_len:]
        except:
            result = "网络错误"
            traceback.print_exc()
        finally:
            return result
        
        
            
    async def _on_danmaku(self, client: blivedm.BLiveClient, message: blivedm.DanmakuMessage):
        print(f'[{client.room_id}] {message.uname}：{message.msg}')
        starttime = time.time()
        #messages = []
        openai.api_key = openai_api_key
        # 设置模型名称
        model_engine = "text-davinci-002"
        question = message.msg
        #item =  {"role": "user", "content": f'"做出尽量简短的回复："+{msg}'}
        log.basicConfig(filename='openai-history.log', level=log.DEBUG)
        #self.history.append({"role": "user", "content": message.msg})
        prompt = self.prepare_input(question)
        self.clip_history()
        #messages = [{"role": "system", "content": set}] + self.history
        #messages =  [ {"role": "system", "content": set},
        #        {"role": "user", "content": message.msg}]
        # ChatGPT is powered by gpt-3.5-turbo, OpenAI’s most advanced language model.
        print('generating text')
        #response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        if USE_MODEL == "OPENAI":
            answer = self.send_openai(prompt)
        elif USE_MODEL == "chatglm-6b":
            answer = self.send_chatglm(prompt)
        #answer = '卧槽'
        # 如果answer有换行，则去掉否则会报错
        answer = answer.replace('\n','')
        answer = answer.replace("'",'')
        
        #self.history.append({"role":"assistant","content":answer})
        self.append_answer(answer)
        print('answer is:',answer, 'time cost is:',time.time()-starttime)
        # 设置要合成的文本
        text = answer
        ttsstart = time.time()
        #生成TTS语音
        command = f'edge-tts --voice zh-CN-XiaoyiNeural --text "{answer}" --write-media output.mp3'  # 将 AI 生成的文本传递给 edge-tts 命令
        subprocess.run(command, shell=True)  # 执行命令行指令
        soundbegin = time.time()
        print('TTS cost time:',soundbegin - ttsstart)
        # 初始化 Pygame
        pygame.mixer.init()

        # 加载语音文件
        pygame.mixer.music.load("output.mp3")

        # 播放语音
        pygame.mixer.music.play()

        # 等待语音播放结束
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # 退出临时语音文件
        pygame.mixer.quit()
        print(f"playing sound cost {time.time()-soundbegin}s")
        print(f'response time:{time.time()-starttime}')


#回复礼物
    async def _on_gift(self, client: blivedm.BLiveClient, message: blivedm.GiftMessage):    
        if message.gift_name == '辣条':
            await client.send_danmaku('谢谢你的辣条')
        if message.gift_name == '小电视':
            await client.send_danmaku('谢谢你的小电视')

#回复舰长
    async def _on_buy_guard(self, client: blivedm.BLiveClient, message: blivedm.GuardBuyMessage):
        if message.gift_name == '总督':
            await client.send_danmaku('谢谢你的总督')
        if message.gift_name == '提督':
            await client.send_danmaku('谢谢你的提督')
        if message.gift_name == '舰长':
            await client.send_danmaku('谢谢你的舰长')

#global wsParam

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())

	
