from tqdm import tqdm
import sys
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
import time
import requests
import urllib.request
import pandas as pd
options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])
driver = webdriver.Chrome(options=options)
driver.get("https://ttsfree.com/login") # site url
print(1)
id = driver.find_element_by_xpath('/html/body/section/div/div/div/div/div[1]/div/form/div[1]/div[1]/input')
print(2)
id.send_keys('your_id') # input your id
password = driver.find_element_by_xpath('/html/body/section/div/div/div/div/div[1]/div/form/div[1]/div[2]/input')
password.send_keys('your_password') # input your password
agree = driver.find_element_by_xpath('/html/body/section/div/div/div/div/div[1]/div/form/div[1]/div[3]/div/label/div/ins')
agree.click()
print(3)
login = driver.find_element_by_xpath('/html/body/section/div/div/div/div/div[1]/div/form/div[3]/input')
login.click()
time.sleep(10)
server2 = driver.find_element_by_xpath('/html/body/section[2]/div[2]/form/div[2]/div[1]/ul/li[2]/a')
server2.send_keys(Keys.ENTER)
korean = driver.find_element_by_xpath('/html/body/section[2]/div[2]/form/div[2]/div[1]/div/div[2]/select/option[105]')
korean.click()
print(1)
def TTS(text, name):

    # write TEXT
    elem = driver.find_element_by_name("input_text")
    elem.clear()
    elem.send_keys(text)
    time.sleep(0.5)
    # TTS conversion
    sayit = driver.find_element_by_xpath('/html/body/section[2]/div[2]/form/div[2]/div[2]/a')
    sayit.send_keys(Keys.ENTER)
    time.sleep(1.5)
    mp3 = None
    start = time.time()
    while mp3 == None:
        try:
            mp3 = driver.find_element_by_xpath('/html/body/section[2]/div[2]/form/div[2]/div[2]/div[2]/div[2]/audio/source[2]').get_attribute('src')
        except:
            pass
        if time.time() - start > 20:
            break
    print(time.time() - start, mp3)
    
    if mp3 == None:
        pass
    else:
        opener=urllib.request.build_opener()
        opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.115 Safari/537.36')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(mp3, 'sunhi\\' + name.split('.')[0] + '.mp3')

df = pd.read_csv('transcripts.txt',  sep = '|', header = None)
df.columns = ['name', 'sentence1', 'sentence2']
for i in tqdm(range(len(df))):
    TTS(df['sentence2'][i], df['name'][i])