# selenium 4
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By


service=Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)