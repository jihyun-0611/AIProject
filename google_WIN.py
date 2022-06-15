#Window
from gettext import find
from itertools import count
from tkinter import SCROLL
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

#a = input("검색할 키워드를 입력 : ")
#image_name = input("저장할 이미지 이름 : ")
driver = webdriver.Chrome()
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")
elem = driver.find_element_by_name("q")
elem.send_keys("서핑보드")
elem.send_keys(Keys.RETURN)

SCROLL_PAUSE_TIME=1
#get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    #scroll down to bottom
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
    #wait to load page
    time.sleep(SCROLL_PAUSE_TIME)
    #calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height: 
        try:
            driver.find_elements_by_css_selector(".mye4qd").click()
        except:
            break
    last_height = new_height


images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
count=1
for image in images:
    try:
        image.click()
        time.sleep(2)
        imgUrl = driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute("src")
        urllib.request.urlretrieve(imgUrl, "surfboard"+ str(count)+".jpg")
        count = count+1 
    except:
        pass

driver.close()