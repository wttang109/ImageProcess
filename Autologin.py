# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:26:31 2018

@author: sunny
"""
# element located

# reference：https://www.jianshu.com/p/d7a966ec1189  作者：凝墨洒泪
from selenium import webdriver
import time
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup

# link to browser
browser = webdriver.Chrome('C:\Python34\Scripts\chromedriver.exe')
browser.get("http://www.tc-pdbus.url.tw/index.php?init=1")

# login
time.sleep(1)
name = browser.find_element_by_name("loginname")
name.send_keys("qqqq")
passwd = browser.find_element_by_name("password")
passwd.send_keys("qqqq")
login_button = browser.find_element_by_xpath("//input[@type='submit'][@value='登入']")
login_button.click()

time.sleep(1)
agree_link = browser.find_element_by_link_text("同意")
agree_link.click()
time.sleep(2)

# located frame
browser.switch_to_frame('Umenu')
reservation = browser.find_element_by_link_text("乘客預約")
reservation.click()
time.sleep(6)

# show the rest of reservation
browser.switch_to.default_content()
browser.switch_to_frame("Rbody")
#booking = browser.find_element_by_xpath('//*[@id="today_times"]')
testsoup = BeautifulSoup(browser.page_source, "html.parser")
today_times = testsoup.select('#today_times')[0].get_text()
print('訂車剩餘次數:', today_times)

day = ["2018-06-11(一)","2018-06-13(三)","2018-06-15(五)"]

ch = 5
i = 0 
j = 0

while today_times == '0':

    browser.switch_to.default_content()
    print('Refresh',j)
    j = j + 1
    browser.switch_to_frame('Umenu')
    reservation = browser.find_element_by_link_text("乘客預約")
    reservation.click()
    time.sleep(12)
    try:
        browser.switch_to.default_content()
        browser.switch_to_frame("Rbody")
        testsoup = BeautifulSoup(browser.page_source, "html.parser")
        today_times = testsoup.select('#today_times')[0].get_text()
        print('訂車剩餘次數:', today_times)
    except:
        browser.switch_to.alert.accept() # confirm alert window
        time.sleep(2)
        print('--確認提示框--')
        browser.switch_to.default_content()
        browser.switch_to_frame("Rbody")
        testsoup = BeautifulSoup(browser.page_source, "html.parser")
        today_times = testsoup.select('#today_times')[0].get_text()
        print('訂車剩餘次數:', today_times)
        
else:
    while today_times != '0':
        #'''    
        #星期日訂一三五
#        t = time.strftime("%Y-%m-%d", time.localtime())
#        today = datetime.date.today()
#        print('今天為:',today)
#        booking1 = today + datetime.timedelta(days=7)
#        print('booking1=',booking1)
        s1 = Select(browser.find_element_by_id('date_prior'))  #實例化Select
#        s1.select_by_value("2018-06-06")
        s1.select_by_visible_text(day[i])
        print('開始訂',day[i])
        time.sleep(4)
#        date_prior = testsoup.select('#date_prior')[0].get_text()
#        print('date_prior=',date_prior)
#        print(time.strftime("%Y-%m-%d", time.localtime()))#年月日
        try:
            browser.switch_to.alert.accept()
            time.sleep(2)
            txt1 = browser.switch_to.alert.text() 
            print('txt1=',txt1)
            time.sleep(2)

        except:
            time.sleep(2)

        s2 = Select(browser.find_element_by_id('time_start'))
        s2.select_by_index(8)
        s4 = Select(browser.find_element_by_id('sele_end_point'))
        s4.select_by_index(1)
        s4.select_by_index(0)
#        cmd = browser.find_element_by_id('cmd_save')
#        cmd.click()
        print('已訂',day[i])
        print('=======================')
        time.sleep(2)
        i = i + 1

#soup  = BeautifulSoup(something, 'lxml')
#plaintext = soup.select('li')[0].get_text().strip()
    else:
        print('訂車完成')    
#        testsoup = BeautifulSoup(browser.page_source, "html.parser")
#        day_week = testsoup.select('#day_week')[0].get_text()   
#        day = browser.find_element_by_id('day_week')
#        day_week_strip = day_week.strip('><')
'''
        print(type(day_week))
        print('day_week=',day_week)
        
        while True:
            if  day_week in book135:
                print('開始訂',day_week)
                ch = ch + 1
                s2 = Select(browser.find_element_by_id('time_start'))
                s2.select_by_index(8)
                s4 = Select(browser.find_element_by_id('sele_end_point'))
                s4.select_by_index(1)
                s4.select_by_index(0)
                #cmd = browser.find_element_by_id('cmd_save')
                #cmd.click()
                time.sleep(2)
                try:
                    browser.switch_to.alert.accept()
                    print(book135[i], end='')
                    print('訂位候補')
                except:
                    print(book135[i], end='')
                    print('訂位成功')
            i = i + 1
            
        else:
            ch = ch + 1
            browser.switch_to.default_content()
            print('Refresh')
            browser.switch_to_frame('Umenu')
            reservation = browser.find_element_by_link_text("乘客預約")
            reservation.click()
            time.sleep(4)
            browser.switch_to.default_content()
            browser.switch_to_frame("Rbody")
            try:
                browser.switch_to.alert.accept()
            except:
                print('No Alert 2')
            
        con = con - 2
        print('剩餘次數:', con)

    else:
        print('Done')

        #星期一訂二四
        s1 = Select(browser.find_element_by_id('date_prior'))  # 實例化Select
        s1.select_by_index(8)
        s2 = Select(browser.find_element_by_id('time_start'))
        s2.select_by_index(8)
        s3 = Select(browser.find_element_by_id('time_end'))
        s3.select_by_index(3)    
        s4 = Select(browser.find_element_by_id('sele_end_point'))
        s4.select_by_index(1)
        #cmd = browser.find_element_by_id('cmd_save')
        #cmd.click()
        time.sleep(2)
'''


