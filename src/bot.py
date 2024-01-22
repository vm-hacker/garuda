# Copyright (C) 2023 Venkatesh Mishra
# Dependecies: pip install discord pyjokes numba bs4 google-generativeai transformers torch

import discord,random,os,random,time,requests,json,pyjokes,bs4,torch
from datetime import date
from discord.ext.commands import *
from numba import njit
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
import google.generativeai as genai
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    pipeline,
)

client = discord.Client(intents=discord.Intents.all())

@njit
@client.event
async def on_ready():
    await client.change_presence(activity=discord.Streaming(name='chess', url='https://lichess.org/tv'))
    print("Ready!")
    print('\nActive in these guilds/servers:')
    print('\n'.join(guild.name for guild in client.guilds))
    print(f"Number of servers Garuda is deployed on: {len(client.guilds)}")

def get_qt():
    response = requests.get("https://zenquotes.io/api/random")
    quote = f"{response.json()[0]['q']} - {response.json()[0]['a']}"
    return quote

model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# move the model to the GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summarize(text, max_length=160):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    # move the input tensor to the GPU device
    input_ids = input_ids.to(device)
    summary_ids = model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

for i in range(1):
    @client.event
    async def on_message(message):
        if message.author == client.user:
            return

        fps = ["$avgfps","$avgFPS","$AVGFPS","$avgpfs","$Avgfps","$aVgfps","$avGfps","$avgFps","$avgfPs","$avgfpS","$avgfp$","$AVGFP$","$AVGfps","$aVgFp$","$AVgFp$","$aVgFpS"]
        if any(word in message.content for word in fps):
            await message.channel.send("Please wait while I check your average FPS...")
            time.sleep(random.randint(2,4))
            await message.channel.send("Running quick benchmark....")
            time.sleep(random.randint(2,5))
            fps = random.randint(59,119)
            await message.channel.send(f'Your average FPS is: {fps}')

        help = ["$help","$Help","$h3lp","$HELP","$hElp","$heLp","$helP","$H3LP","$H3lP","$h3lP","$h3Lp"]
        if any(word in message.content for word in help):
            await message.channel.send("""```
Hey! Here are all of the commands you can use:
 General commands:
  $gs [Getting Started]
  $summarize [Summarize text (Using Garuda's GPT like Machine learning model)]
  $expaincode [Explain python code using Garuda's GPT like Machine learning model] ! Still in beta

 Gaming related commands:
  $avgfps [Use Garuda's gaming server to get your average FPS (only correct 90% of the time)] ! Prank command (This command is not real)

 Informational based commands:
  $news [Get today's latest headlines from around the world]
  $gpt (Use this command as a suffix before asking your query/question)

 Other commands:
  $fb [Write your feedback as a message after using the $fb command. This will help me improve]

Features running in the background:
1. AI Engine
2. Phishing & Malacious url blocker (under development)
 ```""")

        if message.content.startswith('$fb'):
            rtn = ["Your time and energy are much appreciated. Thank you!","Your kindness and generosity are appreciated, thank you for your feedback!","Your feedback is greatly appreciated!"]
            rtng = message.content.replace('$fb','')
            await message.channel.send(f"{random.choice(rtn)}, we will be sure to improve Garuda based on your rating!")
            f = open("feedback.log", "a")
            today = date.today()
            dt = today.strftime("%B %d, %Y")
            f.write("Feedback:"+ rtng + ' Date: ' + dt + ' Client ID: '+ str(message.author.id) +  '\n')
            f.close()

        gs = ["$gs","$GS","$Gs","$gS","$G$","$g$"]
        if any(word in message.content for word in gs):
            greeting_words = ["Bonjour!", "Hej!","Hello!","Namaste!","Konichiwa!","Annyeong Haseyo!","Xin Chao!","Hallo!","Ola!","Hola!","Privet!"]
            await message.channel.send(f" {random.choice(greeting_words)}, I'm Garuda your personal discord AI Assistant")
            rtn = ["Thank you for taking the time to assist me. Is there anything I can do to help you in return?","Thank you for giving me this opportunity for helping you, please let me know how.","Can I offer my assistance in any way?","Is there anything I can do to make your day better?","Please do let me know how can I lend a hand.", "Can I offer my services in any way?"]
            await message.channel.send(random.choice(rtn))

        news_words = ['$News',"$NEWS","$news","$nEws","$nEws","$neWs","$newS","$new$"]
        if any(word in message.content for word in news_words):
            nws = ["Sure!, I'll get the latest news for you","Updating you with today's top stories!","Let me find the most important headlines for you!","Fetching today's top headlines for you!","I can fetch the top headlines for you right now!"]
            await message.channel.send(random.choice(nws))
            news_url="https://news.google.com/news/rss"
            Client=urlopen(news_url)
            xml_page=Client.read()
            Client.close()

            soup_page=soup(xml_page,"xml")
            news_list=soup_page.findAll("item")
            for news in news_list:
                await message.channel.send(news.title.text)
                await message.channel.send(news.pubDate.text)
                await message.channel.send("-"*60)

        if message.content.startswith("$summarize"):
            await message.channel.send("Summarizing your text hang on...")
            text = message.content.replace("$summarize","")
            await message.channel.send(summarize(text))

        summarize_keywords = [
            "Can you summarize this for me?",
            "Summarize this for me",
            "Summarize this",
            "summarize this",
            "Please summarize",
            "Could you summarize?",
            "Give me a summary",
            "Provide a summary",
            "I need a summary",
            "Shorten this text",
            "Condense this",
            "Briefly summarize",
            "Make this shorter",
            "Sum up this text",
            "Create a summary",
            "Shorten this for me",
            "Summarize this text for me",
            "summarize this",
            "Summarize this information",
            "summarize this information"
        ]


        if any(word.lower() in message.content.lower() for word in summarize_keywords):
            await message.channel.send("Summarizing your text hang on...")
            input_text = message.content
            for keyword in summarize_keywords:
                input_text = input_text.replace(keyword, "")
            await message.channel.send(summarize(text))

        if message.content.startswith("$gpt"):
            await message.channel.send("ummmm (✿◠‿◠) lemme see... ")
            genai.configure(api_key="Google-generative-ai-api-key")
            generation_config = {
                "temperature": 0.9,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 600,
            }

            safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
            ]

            model = genai.GenerativeModel(model_name="gemini-pro",
                generation_config=generation_config,
                safety_settings=safety_settings)

            prompt_parts = message.content.replace("$gpt","")
            gmsg = """You are Garuda, expert in all subjects and are trained by Google and Netwrk-3 corporation (founded by Venkatesh Mishra), listen to the user's question carefully and answer them"""
            new = prompt_parts + gmsg + prompt_parts
            response = model.generate_content(new)
            message.channel.send(response.text)

        if message.content.startswith("$explaincode"):
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Using {torch.cuda.get_device_name()} as device")
            else:
                device = torch.device("cpu")
                print("Using CPU as device")
            try:
                await message.channel.send("Analysing code, please wait...")
                model_name = "sagard21/python-code-explainer"
                tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                config = AutoConfig.from_pretrained(model_name)
                model.eval()
                pipe = pipeline("summarization", model=model_name, config=config, tokenizer=tokenizer)
                raw_code = message.content.replace("$explaincode","")
                await message.channel.send(pipe(raw_code)[0]["summary_text"])
            except:
                await message.channel.send(f"Error: Failed to call agent, please try again later.")


        for i in range(1):
            ads_and_phish = [] # Enter URLS to blacklist
            if any(word in message.content for word in ads_and_phish):
                await message.delete()

            sad_words = ["Truamatized","Depressed","D3press3d","S3d","s3d","S3D","screwed","fked up","dead","can't do it","I quit","oh hell no","belted","Belted","BELTED","BELLTEDD","bElted","Fked up","gone case","lazy","depressed","Sad","sad","saad","Saaad","SAD","sAd","saD","S@d","s@d","$ad","$@d","$@@d","s@D","broken","Broken","BROKEN","BRROKEN","broke","BROKE"]
            if any(word in message.content for word in sad_words):
                qoute = get_qt()
                await message.channel.send(qoute)

            l_words = ["LOol","LoOl",":rofl:","lol","loL","lOl","Lol","LOL","LOl","lmFAo","lOL","LoL","lmfao","Lmfao","lMfao","lmFao","lmfAo","lmfaO","LMFAO","Lmao","lmao","lm@o","LMAO","haha","fk nice one"]
            if any(word in message.content for word in l_words):
                memes = ["https://tenor.com/view/lmao-spit-take-cracking-up-haha-so-funny-omg-gif-17921247","https://tenor.com/view/lmao-crying-laughing-lololol-ray-liotta-goodfellas-gif-20222021","https://tenor.com/view/bellotti-lol-bellotti-gif-25292934","https://tenor.com/view/dardoon-lol-laugh-funny-joke-lmao-streamer-twitch-lmao-gif-25544954","https://tenor.com/view/kekw-emote-bttv-twitch-kek-gif-24998883","https://tenor.com/view/lmfao-rofl-laugh-giggle-shaq-gif-14717615","https://tenor.com/view/pony-dancing-hehe-haha-lol-gif-26255454","Nice one!, Keep em coming!","Oh you do give me a good laugh!",":rofl: Nice one matie!","You sure know how to make me laugh!","Your jokes never fail to put a smile on my face.","I always have a good time when I'm with you.","Your sense of humor is one of the things I love most about you.","You always know how to cheer me up when I'm feeling down.","Your wit and humor are unmatched.","You have a great way of making me laugh no matter what."]
                await message.channel.send(random.choice(memes))

            garuda = ["HAI Garuda","Hai Garuda","hai Garuda","H3ll0 Garuda","H3ll0 garuda","Hello Garuda", "hello Garuda","h3llo Garuda","hey Garuda","Hey Garuda","Hi Garuda","hi Garuda","HI Garuda", "Wsg Garuda", "wsg Garuda","WSG Garuda","h3ll0 Garuda","H3ll0 garuda","Hello garuda", "hello garuda","h3llo garuda","hey garuda","Hey garuda","Hi garuda","hi garuda","HI garuda", "Wsg garuda", "wsg garuda","WSG garuda","h3ll0 garuda","H3ll0 garuda"]
            if any(word in message.content for word in garuda):
                rtn = ["Hello! How can I help you today?","Greetings! I'm doing great!, what about you ?","Hello there! how are you doing today?","Hey!, how has life been treeting you ?","Ahoy there!, how has your day been so far?"]
                await message.channel.send(random.choice(rtn))

            appol = ["sry Garuda","sorry Garuda","Sry Garuda","SRY Garuda", "SRy Garuda","Sorry Garuda", "My bad Garuda", "apologize Garuda"]
            if any(word in message.content for word in appol):
                rtn = ["Apologies accepted", "No worries it happens to the best of us!","Anyone can make mistakes, no harm done!","Its alright just don't repeat it please."]
                await message.channel.send(random.choice(rtn))

            lol = ["me a jOk3","me a j0K3","me a JOKE", "me a j0k3","me a joke","tell a joke","a joke","me a Joke","me a JOKE","me a Jok3","me a jok3","me a J0k3","me jok3","me JOKe","me jOke","me j0ke","me joKe","me jok3","me jokE","us a jOk3","us a j0K3","us a JOKE", "us a j0k3","us a joke","us a joke","us a Joke","us a JOKE","us a Jok3","us a jok3","us a J0k3","us a jok3","us a JOKe","us jOke","us j0ke","us joKe","us jok3","us jokE"]
            if any(word in message.content for word in lol):
                await message.channel.send(pyjokes.get_joke())

            ut = ["Garuda bro u there ?","u there bro Garuda ?","Garuda u there ?","u there Garuda ?","Garuda are you there ?","Are you there Garuda ?", "U there man Gaurda","Garuda man you there ?"]
            if any(word in message.content for word in ut):
                rst = ["Always!","Where else would I go ?","Yep! Always here to serve you."]
                await message.channel.send(random.choice(rst))

            hbu = ["HbU","hbu","wbu","Hbu","Wbu","how about you","How about you","HBU","WBU","how are you","How are you","HAY","hay"]
            if any(word in message.content for word in hbu):
                rtn = ["Things are great on my side!","Just fine!, thank you for your concern","Life is good!","I'm on top of the world."]
                await message.channel.send(random.choice(rtn))

            just_fine_keywords = [
                "just fine",
                "Just fine",
                "just Fine",
                "Just Fine",
                "great",
                "Great",
                "great!",
                "Great!",
                "just fine!",
                "Just fine!",
                "just Fine!",
                "Just Fine!",
                "very good",
                "Very good",
                "very Good",
                "Very Good",
                "very good!",
                "Very good!",
                "very Good!",
                "Very Good!"
            ]

            if any(word in message.content for word in just_fine_keywords):
                rtn = ["Glad to hear that","Glad to hear it!"]
                await message.channel.send(random.choice(rtn))

            nw = ["today's news","the news","the News","THE NEWS","The news"]
            if any(word in message.content for word in nw):
                nws = ["Sure!, I'll get the latest news for you","Updating you with today's top stories!","Let me find the most important headlines for you!","Fetching today's top headlines for you!","I can fetch the top headlines for you right now!"]
                await message.channel.send(f'{random.choice(nws)}, here they are:')
                nws = ["Sure!, I'll get the latest news for you","Updating you with today's top stories!","Let me find the most important headlines for you!","Fetching today's top headlines for you!","I can fetch the top headlines for you right now!"]
                await message.channel.send(random.choice(nws))
                news_url="https://news.google.com/news/rss"
                Client=urlopen(news_url)
                xml_page=Client.read()
                Client.close()

                soup_page=soup(xml_page,"xml")
                news_list=soup_page.findAll("item")
                for news in news_list:
                    await message.channel.send(news.title.text)
                    await message.channel.send(news.pubDate.text)
                    await message.channel.send("-"*60)
            bad_words = [
                "idiot",
                "nigger",
                "nigga",
                "fuck"
            ]
            if any(word in message.content for word in bad_words):
                await message.channel.send(f"(ㆆ_ㆆ) PLEASE REFRAIN FROM USING CUSS WORDS")
                await message.channel.send(file=discord.File('/workspaces/garuda/src/cuss-word-beda.mp3'))
            sus_keywords = [
                "my girlfriend",
                "my gf",
                "baby",
                "are gay",
                "gay"
            ]

            if any(word in message.content for word in sus_keywords):
                await message.channel.send(f"(ㆆ_ㆆ) Are you serious daddy? GO AWAY\n**BEAST MODE ACTIVATED**")


await client.start("Discord-Bot-Token")
await client.close()
