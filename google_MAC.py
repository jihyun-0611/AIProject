#MAC
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import urllib.request

driver = webdriver.Chrome("/Users/takyeji/Desktop/ai/selenium/chromedriver")
driver.get(r"https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl")
elem = driver.find_element(By.NAME, "q") #검색창에
elem.send_keys("fish") #글 자동입력
elem.send_keys(Keys.RETURN) #엔터키

SCROLL_PAUSE_TIME = 1

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # 로딩시간
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        try:
            driver.find_element_by_css_selector(".mye4qd").click() #결과더보기버튼 클릭
        except: 
            break
    last_height = new_height
    
images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd") 
count = 1
for image in images: #이미지들중 하나씩 뽑아
    try:
        image.click() #다운할 이미지 클릭해
        time.sleep(3) #3초 대기
        imgUrl = driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute("src")
        #이미지가 다 받아지면 url을 찾아 다운
        urllib.request.urlretrieve(imgUrl, "fish" + str(count) + ".jpg") 
        count = count + 1 #이미지 다운시마다 1씩 증가
    except:
        pass

driver.close()